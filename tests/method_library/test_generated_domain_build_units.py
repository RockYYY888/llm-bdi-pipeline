from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from domain_model.materialization import write_masked_domain_file
from method_library.synthesis.errors import HTNSynthesisError, LLMStreamingResponseError
from method_library.validation.validator import MethodLibraryValidator
from method_library.validation.minimal_validation import validate_domain_complete_coverage
from method_library.synthesis.prompts import (
	build_domain_htn_system_prompt,
	build_domain_prompt_analysis_payload,
	build_domain_htn_user_prompt,
)
from method_library.synthesis.domain_prompts import _render_method_blueprint_blocks
from method_library.synthesis.schema import HTNLiteral, HTNMethodLibrary, HTNMethodStep
from method_library.synthesis.synthesizer import HTNMethodSynthesizer
from execution_logging.execution_logger import ExecutionLogger
from plan_library import (
	LibraryValidationRecord,
	PlanLibrary,
	PlanLibraryArtifactBundle,
	PlanLibraryGenerationPipeline,
	TranslationCoverage,
	load_plan_library_artifact_bundle,
	persist_plan_library_artifact_bundle,
)
from tests.support.plan_library_generation_support import (
	DOMAIN_FILES,
	build_method_library_from_domain_file,
)
from utils.hddl_parser import HDDLParser


def test_masked_domain_removes_official_methods_and_preserves_domain_shape(tmp_path: Path) -> None:
	official_domain_file = DOMAIN_FILES["blocksworld"]
	official_domain = HDDLParser.parse_domain(official_domain_file)

	masked = write_masked_domain_file(
		official_domain_file=official_domain_file,
		output_path=tmp_path / "masked_domain.hddl",
	)
	masked_domain = masked["masked_domain"]

	assert masked["original_method_count"] > 0
	assert masked["masked_method_count"] == 0
	assert "(:method" not in masked["masked_domain_text"]
	assert len(masked_domain.actions) == len(official_domain.actions)
	assert len(masked_domain.tasks) == len(official_domain.tasks)
	assert len(masked_domain.predicates) == len(official_domain.predicates)
	assert len(masked_domain.types) == len(official_domain.types)


def test_domain_prompt_is_query_aligned_and_does_not_leak_official_methods(tmp_path: Path) -> None:
	official_domain_file = DOMAIN_FILES["blocksworld"]
	official_domain = HDDLParser.parse_domain(official_domain_file)
	masked = write_masked_domain_file(
		official_domain_file=official_domain_file,
		output_path=tmp_path / "masked_domain.hddl",
	)
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(masked["masked_domain"])
	derived_analysis = build_domain_prompt_analysis_payload(
		masked["masked_domain"],
		action_analysis=analysis,
	)

	system_prompt = build_domain_htn_system_prompt()
	user_prompt = build_domain_htn_user_prompt(
		masked["masked_domain"],
		schema_hint='{"tasks":[...]}',
		action_analysis=analysis,
		derived_analysis=derived_analysis,
		query_sequence=(
			{
				"instruction_id": "query_1",
				"source_text": "Put block b4 on block b2, then put block b1 on block b4.",
				"problem_file": "p01.hddl",
			},
		),
		temporal_specifications=(
			{
				"instruction_id": "query_1",
				"source_text": "Put block b4 on block b2, then put block b1 on block b4.",
				"ltlf_formula": "do_put_on(b4, b2) & X(do_put_on(b1, b4))",
				"referenced_events": [
					{"event": "do_put_on", "arguments": ["b4", "b2"]},
					{"event": "do_put_on__e2", "arguments": ["b1", "b4"]},
				],
			},
		),
	)

	for official_method in official_domain.methods:
		assert official_method.name not in system_prompt
		assert official_method.name not in user_prompt

	assert "problem.hddl" not in user_prompt
	assert "target_literals" not in user_prompt
	assert "target_task_bindings" not in user_prompt
	assert "Do not condition the library on any benchmark query." not in user_prompt
	assert "<query_sequence>" not in user_prompt
	assert "<temporal_specifications>" in user_prompt
	assert "Put block b4 on block b2" not in user_prompt
	assert "source_instruction_ids" in user_prompt
	assert "primitive_action_schemas:" in user_prompt
	assert "declared_compound_tasks:" in user_prompt
	assert "<structural_contract>" in user_prompt
	assert "<causal_executability>" in user_prompt
	assert (
		"required_method_task_names: "
		f"{json.dumps([task.name for task in masked['masked_domain'].tasks])}"
		in user_prompt
	)
	assert "method_blueprints" in user_prompt
	assert '"method_family_schemas"' in user_prompt
	assert '"uncovered_prerequisite_families"' in user_prompt
	assert '"primitive_action"' in user_prompt
	assert "Preserve distinct AUX witness roles" in user_prompt
	assert "Give each variable one declared type" in system_prompt
	assert "aligned with the temporal specifications" in system_prompt
	assert "primitive and compound subtasks are not swapped" in user_prompt
	assert "Primitive action names from primitive_action_schemas" in user_prompt
	assert "Actions are operators, not predicates." in system_prompt
	assert "M is domain-complete" in system_prompt
	assert "Constructive methods must be causally executable" in system_prompt
	assert "Do not copy object constants from temporal specifications into M" in system_prompt
	assert "If a method has zero or one subtask, ordering must be []." in system_prompt
	assert 'local pairwise ordering edges only: [["s1", "s2"]].' in system_prompt
	assert "must be quoted strings" in system_prompt
	assert "Every positive dynamic precondition" in user_prompt
	assert "A direct primitive leaf is valid only if" in user_prompt
	assert "Bind task parameters according to action effect argument positions" in user_prompt
	assert "Do not require mutually exclusive fluent values in one context" in user_prompt
	assert "Methods that are not already satisfied must contain real subtasks" in user_prompt
	assert "primitive leaf methods must include the primitive action itself" in user_prompt
	assert "Use temporal_specifications as the only task-level supervision" in user_prompt
	assert "keeping methods reusable and variable-parameterized" in user_prompt
	assert "from temporal_specifications" in user_prompt
	assert "do not drop tasks absent from temporal_specifications" in user_prompt
	assert "methods.task_name covers every required_method_task_names entry" in user_prompt
	assert "each ordering edge [before, after] must use two distinct local_step_ids" in user_prompt
	assert "methods with fewer than two subtasks have empty ordering" in user_prompt
	assert "local step ids.\n- methods with fewer than two subtasks" in user_prompt
	assert "local step ids.- methods" not in user_prompt
	assert "every primitive action precondition is supported" in user_prompt
	assert "support subgoals precede primitive actions that need their effects" in user_prompt
	assert "Before emitting JSON, check that:" in user_prompt
	assert 'ordering must be an array of two-element step-id arrays such as [["s1", "s2"]].' in user_prompt
	assert "never emit bare tokens" in user_prompt
	assert "primitive_actions:" not in user_prompt
	assert "<silent_self_check>" not in user_prompt
	assert "Emit one JSON object with keys compound_tasks and methods." in user_prompt


def test_domain_prompt_analysis_exposes_composition_and_acquisition_families(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["blocksworld"],
		output_path=tmp_path / "masked_blocksworld.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	contracts = {
		str(contract["task_name"]): contract
		for contract in payload["domain_task_contracts"]
	}
	do_move_contract = contracts["do_move"]

	assert do_move_contract["headline_candidates"] == ["on"]
	assert any(
		"do_put_on(?x, ?y) stabilizes on" in line
		for line in do_move_contract["composition_support_tasks"]
	)
	assert any(
		"holding(?x) via pick-up(?x)" in line
		for line in do_move_contract["prerequisite_acquisition_templates"]
	)
	assert any(
		"holding(?x) via unstack(?x, AUX_BLOCK1)" in line
		for line in do_move_contract["prerequisite_acquisition_templates"]
	)

	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}
	do_put_on_blueprint = blueprints["do_put_on"]
	do_move_blueprint = blueprints["do_move"]
	do_clear_blueprint = blueprints["do_clear"]

	assert do_put_on_blueprint["headline_candidates"] == ["on"]
	assert any(
		"do_move(?x, ?y) stabilizes on" in line
		for line in do_put_on_blueprint["headline_support_tasks"]
	)
	assert any(
		family.get("final_step") == "stack(?x, ?y)"
		for family in do_move_blueprint["method_family_schemas"]
		if isinstance(family, dict)
	)
	assert any(
		"pick_up(?x)" in line
		for line in do_clear_blueprint["direct_primitive_achievers"]
	)
	assert any(
		"put_down(?x)" in line
		for line in do_clear_blueprint["direct_primitive_achievers"]
	)


def test_domain_prompt_analysis_keeps_transport_blueprints_type_aligned(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["transport"],
		output_path=tmp_path / "masked_transport.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}
	deliver_blueprint = blueprints["deliver"]
	get_to_blueprint = blueprints["get-to"]
	load_blueprint = blueprints["load"]
	unload_blueprint = blueprints["unload"]

	assert not any(
		"drive(?p" in line
		for line in deliver_blueprint["direct_primitive_achievers"]
	)
	assert not any(
		"deliver(" in " ".join(list(family.get("recursive_support_calls") or ()))
		for family in deliver_blueprint["method_family_schemas"]
		if isinstance(family, dict)
	)
	assert get_to_blueprint["direct_primitive_achievers"] == ["none"]
	assert any(
		"drive(?v" in str(family.get("final_step") or "")
		for family in get_to_blueprint["method_family_schemas"]
		if isinstance(family, dict)
	)
	assert load_blueprint["headline_candidates"] == ["helper_only"]
	assert unload_blueprint["headline_candidates"] == ["helper_only"]
	assert load_blueprint["preferred_family_shape"] == "direct_leaf"
	assert unload_blueprint["preferred_family_shape"] == "direct_leaf"
	assert load_blueprint["method_family_schemas"] == ["none"]
	assert unload_blueprint["method_family_schemas"] == ["none"]
	assert "deliver(?p, ?l)" not in deliver_blueprint["support_call_palette"]
	assert "get-to(?v, ?l)" not in get_to_blueprint["support_call_palette"]
	assert any(
		"pick_up(?v, ?l, ?p" in line
		for line in load_blueprint["direct_primitive_achievers"]
	)
	assert any(
		"drop(?v, ?l, ?p" in line
		for line in unload_blueprint["direct_primitive_achievers"]
	)


def test_domain_prompt_analysis_recovers_satellite_helper_task_families(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["satellite"],
		output_path=tmp_path / "masked_satellite.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}
	activate_instrument_blueprint = blueprints["activate_instrument"]
	auto_calibrate_blueprint = blueprints["auto_calibrate"]

	assert any(
		"switch_on(?ai_i, ?ai_s)" in line
		for line in activate_instrument_blueprint["direct_primitive_achievers"]
	)
	assert any(
		"calibrate(?ac_s, ?ac_i, AUX_CALIB_DIRECTION1)" in line
		for line in auto_calibrate_blueprint["direct_primitive_achievers"]
	)


def test_domain_schema_hint_matches_method_centric_json_schema() -> None:
	schema_hint = HTNMethodSynthesizer()._domain_schema_hint()

	assert "compound_tasks" in schema_hint
	assert "methods" in schema_hint
	assert '"task_name":"TASK"' in schema_hint
	assert '"method_name":"m_task_constructive"' in schema_hint


def test_rendered_method_blueprints_use_compact_prompt_shape() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["transport"])
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	deliver_blueprint = next(
		blueprint
		for blueprint in payload["method_blueprints"]
		if blueprint["task_name"] == "deliver"
	)

	rendered = json.loads(_render_method_blueprint_blocks([deliver_blueprint]))[0]

	assert rendered["task"].startswith("deliver(")
	assert "task_signature" not in rendered
	assert "typed_task_signature" not in rendered
	assert rendered["headline"] == "at"
	assert isinstance(rendered["uncovered_prerequisite_families"], list)
	assert isinstance(rendered["uncovered_prerequisite_families"][0], dict)
	assert "need" in rendered["uncovered_prerequisite_families"][0]
	assert "primitive_action" in rendered["uncovered_prerequisite_families"][0]
	assert rendered["uncovered_prerequisite_families"][0]["support_kind"] == "primitive_support"
	assert "support_call_palette" not in rendered


def test_rendered_helper_only_transport_leaf_tasks_omit_fake_headlines(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["transport"],
		output_path=tmp_path / "masked_transport.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}

	rendered = json.loads(
		_render_method_blueprint_blocks([blueprints["load"], blueprints["unload"]]),
	)

	for entry in rendered:
		assert "headline" not in entry
		assert "headlines" not in entry
		assert entry["preferred_family_shape"] == "direct_leaf"
		assert "direct_primitive_achievers" in entry
		assert all("primitive_action" in item for item in entry["direct_primitive_achievers"])


def test_library_postprocess_bound_helpers_are_callable_via_synthesizer_instance() -> None:
	synthesizer = HTNMethodSynthesizer()

	assert synthesizer._domain_schema_hint().startswith('{"compound_tasks"')
	assert synthesizer._looks_like_variable("ARG1")
	assert synthesizer._looks_like_variable("?x")


def test_method_synthesis_transport_preserves_configured_output_budget() -> None:
	synthesizer = HTNMethodSynthesizer(model="deepseek-v4-pro")

	assert synthesizer._apply_method_synthesis_provider_token_ceiling(None) is None
	assert synthesizer._apply_method_synthesis_provider_token_ceiling(144000) == 144000


def test_method_synthesis_openai_compatible_profile_uses_configured_output_budget() -> None:
	synthesizer = HTNMethodSynthesizer(
		model="deepseek-v4-pro",
		base_url="https://api.deepseek.com",
		max_tokens=144000,
	)

	profile = synthesizer._method_synthesis_request_profile(
		prompt={"system": "x", "user": "y"},
	)

	assert profile["name"] == "openai_compatible_json_chat"
	assert profile["stream_response"] is False
	assert profile["completion_max_tokens"] == 144000
	assert profile["max_tokens_policy"] == "configured_method_synthesis_max_tokens"
	assert profile["thinking_type"] == "enabled"
	assert profile["reasoning_effort"] == "max"


def test_method_synthesis_transport_uses_openai_compatible_json_request() -> None:
	captured_kwargs = {}

	class FakeCompletions:
		def create(self, **kwargs):
			captured_kwargs.update(kwargs)
			return object()

	class FakeChat:
		def __init__(self):
			self.completions = FakeCompletions()

	class FakeClient:
		def __init__(self):
			self.chat = FakeChat()

	synthesizer = HTNMethodSynthesizer(
		model="deepseek-v4-pro",
		base_url="https://api.deepseek.com",
	)
	synthesizer.client = FakeClient()

	synthesizer._create_chat_completion({"system": "x", "user": "y"}, max_tokens=144000)

	assert captured_kwargs["stream"] is False
	assert captured_kwargs["response_format"] == {"type": "json_object"}
	assert captured_kwargs["max_tokens"] == 144000
	assert captured_kwargs["reasoning_effort"] == "max"
	assert captured_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_method_synthesis_transport_create_phase_has_wall_clock_guard() -> None:
	class SlowCompletions:
		def create(self, **kwargs):
			_ = kwargs
			time.sleep(0.05)
			return object()

	class FakeChat:
		def __init__(self):
			self.completions = SlowCompletions()

	class FakeClient:
		def __init__(self):
			self.chat = FakeChat()

	synthesizer = HTNMethodSynthesizer(
		model="other/model",
		base_url="https://api.example.com/v1",
		timeout=0.05,
	)
	synthesizer.client = FakeClient()

	with pytest.raises(TimeoutError, match="wall-clock timeout"):
		synthesizer._create_chat_completion(
			{"system": "x", "user": "y"},
			max_tokens=16,
			request_profile={
				"name": "openai_compatible_json_chat",
				"stream_response": False,
				"first_chunk_timeout_seconds": 0.01,
				"thinking_type": "enabled",
				"reasoning_effort": "max",
			},
			request_timeout_seconds=0.01,
		)


def test_method_synthesis_transport_enforces_wall_clock_timeout() -> None:
	class SlowSynthesizer(HTNMethodSynthesizer):
		def _call_llm_direct(
			self,
			prompt,
			*,
			max_tokens=None,
			transport_metadata=None,
			request_profile=None,
			request_timeout_seconds=None,
		):
			if transport_metadata is not None:
				transport_metadata["llm_request_id"] = "req_timeout"
				transport_metadata["llm_response_mode"] = "non_streaming"
				transport_metadata["llm_request_profile"] = dict(request_profile or {}).get("name")
			time.sleep(0.05)
			return "{}", "stop", dict(transport_metadata or {})

	synthesizer = SlowSynthesizer(model="deepseek-v4-pro", timeout=0.01, max_tokens=144000)

	with pytest.raises(TimeoutError, match="configured timeout") as exc_info:
		synthesizer._call_llm({"system": "x", "user": "y"}, max_tokens=16)

	assert getattr(exc_info.value, "transport_metadata", {}) == {
		"llm_request_id": "req_timeout",
		"llm_response_mode": "non_streaming",
		"llm_request_profile": "openai_compatible_json_chat",
		"llm_first_chunk_timeout_seconds": 0.0,
		"llm_completion_max_tokens": 144000,
		"llm_max_tokens_policy": "configured_method_synthesis_max_tokens",
		"llm_thinking_type": "enabled",
		"llm_reasoning_effort": "max",
		"llm_request_timeout_seconds": 0.01,
	}


def test_method_synthesis_transport_streaming_captures_request_id_and_timings() -> None:
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
		def __init__(self):
			self.closed = False

		def __iter__(self):
			yield FakeChunk("req_stream_123", '{"compound_tasks":[]')
			yield FakeChunk("req_stream_123", ',"methods":[]}', finish_reason="stop")

		def close(self):
			self.closed = True

	synthesizer = HTNMethodSynthesizer()

	response_text, finish_reason, transport_metadata = synthesizer._consume_streaming_llm_response(
		FakeStream(),
		transport_metadata={},
	)

	assert response_text == '{"compound_tasks":[],"methods":[]}'
	assert finish_reason == "stop"
	assert transport_metadata["llm_request_id"] == "req_stream_123"
	assert transport_metadata["llm_response_mode"] == "streaming"
	assert transport_metadata["llm_first_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_stream_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_content_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_complete_json_seconds"] >= 0.0


def test_method_synthesis_transport_counts_empty_stream_chunk_before_content() -> None:
	class FakeDelta:
		def __init__(self, content=None):
			self.content = content

	class FakeChoice:
		def __init__(self, content=None, finish_reason=None):
			self.delta = FakeDelta(content)
			self.finish_reason = finish_reason

	class FakeChunk:
		def __init__(self, chunk_id, content=None, finish_reason=None):
			self.id = chunk_id
			self.choices = [FakeChoice(content, finish_reason=finish_reason)]

	class FakeStream:
		handshake_seconds = 1.25

		def __iter__(self):
			yield FakeChunk("req_empty_first", "")
			yield FakeChunk("req_empty_first", '{"compound_tasks":[],"methods":[]}', "stop")

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()

	response_text, finish_reason, transport_metadata = synthesizer._consume_streaming_llm_response(
		FakeStream(),
		transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
		total_timeout_seconds=1.0,
	)

	assert response_text == '{"compound_tasks":[],"methods":[]}'
	assert finish_reason == "stop"
	assert transport_metadata["llm_stream_handshake_seconds"] == 1.25
	assert transport_metadata["llm_first_stream_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_chunk_seconds"] == transport_metadata[
		"llm_first_stream_chunk_seconds"
	]
	assert transport_metadata["llm_first_content_chunk_seconds"] >= 0.0


def test_method_synthesis_transport_enforces_first_chunk_deadline_during_stream_consumption() -> None:
	class BlockingStream:
		def __iter__(self):
			return self

		def __next__(self):
			time.sleep(0.05)
			raise AssertionError("stream iteration should have timed out before yielding")

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(TimeoutError, match="first-chunk deadline") as exc_info:
		synthesizer._consume_streaming_llm_response(
			BlockingStream(),
			transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
			total_timeout_seconds=0.1,
		)

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_response_mode"] == "streaming"
	assert transport_metadata["llm_first_chunk_timeout_seconds"] == 0.01
	assert transport_metadata.get("llm_first_chunk_seconds") is None


def test_method_synthesis_transport_ignores_reasoning_payload_without_storing_it() -> None:
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
				return FakeChunk("req_reasoning_stream", reasoning="thinking")
			time.sleep(0.05)
			raise AssertionError("stream iteration should have timed out by total deadline")

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(TimeoutError, match="configured timeout") as exc_info:
		synthesizer._consume_streaming_llm_response(
			ReasoningThenBlockingStream(),
			transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
			total_timeout_seconds=0.02,
		)

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_request_id"] == "req_reasoning_stream"
	assert transport_metadata["llm_first_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_stream_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_reasoning_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_reasoning_chunks_ignored"] == 2
	assert "llm_reasoning_preview" not in transport_metadata
	assert "llm_reasoning_characters" not in transport_metadata
	assert "llm_first_content_chunk_seconds" not in transport_metadata


def test_method_synthesis_streaming_total_timeout_survives_off_main_thread() -> None:
	class FakeDelta:
		content = None
		reasoning = "thinking"

	class FakeChoice:
		delta = FakeDelta()
		finish_reason = None

	class FakeChunk:
		id = "req_off_main_timeout"
		choices = [FakeChoice()]

	class ReasoningThenBlockingStream:
		def __init__(self):
			self.index = 0

		def __iter__(self):
			return self

		def __next__(self):
			self.index += 1
			if self.index == 1:
				return FakeChunk()
			time.sleep(0.05)
			raise AssertionError("deadline timer should convert this into a timeout")

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()
	result: dict[str, BaseException] = {}

	def consume_stream() -> None:
		try:
			synthesizer._consume_streaming_llm_response(
				ReasoningThenBlockingStream(),
				transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
				total_timeout_seconds=0.02,
			)
		except BaseException as exc:
			result["exception"] = exc

	thread = threading.Thread(target=consume_stream)
	thread.start()
	thread.join(timeout=1.0)

	assert not thread.is_alive()
	exception = result.get("exception")
	assert isinstance(exception, TimeoutError)
	transport_metadata = getattr(exception, "transport_metadata", {})
	assert transport_metadata["llm_request_id"] == "req_off_main_timeout"
	assert transport_metadata["llm_reasoning_chunks_ignored"] == 1


def test_method_synthesis_retries_stream_failures_with_same_profile() -> None:
	class RetryingSynthesizer(HTNMethodSynthesizer):
		def __init__(self):
			super().__init__(model="deepseek-v4-pro")
			self.call_count = 0

		def _call_llm(self, prompt, *, max_tokens=None):
			self.call_count += 1
			error = LLMStreamingResponseError(
				"LLM response did not contain usable textual JSON content. finish_reason='length'",
				finish_reason="length",
			)
			error.transport_metadata = {
				"llm_request_id": f"req_retry_{self.call_count}",
				"llm_response_mode": "non_streaming",
				"llm_request_profile": "openai_compatible_json_chat",
			}
			raise error

	synthesizer = RetryingSynthesizer()
	metadata = {}
	with pytest.raises(HTNSynthesisError, match="LLM request failed"):
		synthesizer._request_complete_llm_library(
			{"system": "x", "user": "y"},
			domain=type("FakeDomain", (), {"actions": [], "tasks": [], "predicates": []})(),
			metadata=metadata,
			max_tokens=256,
		)

	assert synthesizer.call_count == 6
	assert metadata["llm_attempts"] == 6
	assert metadata["llm_generation_attempts"] == 6
	assert [attempt["request_id"] for attempt in metadata["llm_attempt_trace"]] == [
		"req_retry_1",
		"req_retry_2",
		"req_retry_3",
		"req_retry_4",
		"req_retry_5",
		"req_retry_6",
	]
	assert all(
		attempt["request_profile"] == "openai_compatible_json_chat"
		for attempt in metadata["llm_attempt_trace"]
	)


def test_method_synthesis_retries_then_accepts_successful_response() -> None:
	class EventuallySuccessfulSynthesizer(HTNMethodSynthesizer):
		def __init__(self):
			super().__init__(model="deepseek-v4-pro")
			self.call_count = 0

		def _call_llm(self, prompt, *, max_tokens=None):
			self.call_count += 1
			if self.call_count < 6:
				error = TimeoutError("first attempt timed out")
				error.transport_metadata = {
					"llm_request_id": f"req_retry_{self.call_count}",
					"llm_response_mode": "non_streaming",
					"llm_request_profile": "openai_compatible_json_chat",
				}
				raise error
			return (
				'{"compound_tasks":[],"methods":[]}',
				"stop",
				{
					"llm_request_id": "req_retry_6",
					"llm_response_mode": "non_streaming",
					"llm_request_profile": "openai_compatible_json_chat",
				},
			)

	synthesizer = EventuallySuccessfulSynthesizer()
	metadata = {}
	library, response_text, finish_reason = synthesizer._request_complete_llm_library(
		{"system": "x", "user": "y"},
		domain=type("FakeDomain", (), {"actions": [], "tasks": [], "predicates": []})(),
		metadata=metadata,
		max_tokens=256,
	)

	assert synthesizer.call_count == 6
	assert finish_reason == "stop"
	assert response_text == '{"compound_tasks":[],"methods":[]}'
	assert library.compound_tasks == []
	assert library.methods == []
	assert metadata["llm_attempts"] == 6
	assert metadata["llm_generation_attempts"] == 6
	assert metadata["llm_request_id"] == "req_retry_6"
	assert metadata["llm_attempt_trace"][-1]["request_id"] == "req_retry_6"


def test_method_synthesis_transport_does_not_store_reasoning_when_no_json_arrives() -> None:
	class FakeDelta:
		def __init__(self, reasoning=None):
			self.content = None
			self.reasoning = reasoning

	class FakeChoice:
		def __init__(self, reasoning=None, finish_reason=None):
			self.delta = FakeDelta(reasoning=reasoning)
			self.finish_reason = finish_reason
			self.reasoning = reasoning

	class FakeChunk:
		def __init__(self, chunk_id, reasoning=None, finish_reason=None):
			self.id = chunk_id
			self.choices = [FakeChoice(reasoning=reasoning, finish_reason=finish_reason)]

	class FakeStream:
		def __iter__(self):
			yield FakeChunk(
				"req_reasoning_only",
				reasoning="We need to output JSON with top-level keys compound_tasks and methods.",
			)
			yield FakeChunk(
				"req_reasoning_only",
				reasoning="Now for all empty methods, parameters are the task signature parameters.",
				finish_reason="length",
			)

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(LLMStreamingResponseError) as exc_info:
		synthesizer._consume_streaming_llm_response(FakeStream(), transport_metadata={})

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_request_id"] == "req_reasoning_only"
	assert transport_metadata["llm_response_mode"] == "streaming"
	assert transport_metadata["llm_reasoning_chunks_ignored"] == 4
	assert "llm_reasoning_characters" not in transport_metadata
	assert "llm_reasoning_preview" not in transport_metadata


def test_method_synthesis_transport_returns_raw_json_like_text_for_downstream_salvage() -> None:
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
			yield FakeChunk("req_salvage_123", '{"compound_tasks":[],"methods":[{"method_name":"m1"}]} trailing')
			yield FakeChunk("req_salvage_123", "", finish_reason="stop")

		def close(self):
			return None

	synthesizer = HTNMethodSynthesizer()

	response_text, finish_reason, transport_metadata = synthesizer._consume_streaming_llm_response(
		FakeStream(),
		transport_metadata={},
	)

	assert response_text.startswith('{"compound_tasks":[],"methods":')
	assert finish_reason == "stop"
	assert transport_metadata["llm_request_id"] == "req_salvage_123"


def test_library_postprocess_normalises_typed_method_parameter_surfaces() -> None:
	synthesizer = HTNMethodSynthesizer()
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = synthesizer._parse_llm_library(
			'{"compound_tasks":[{"name":"do_put_on","parameters":["?x:block","?y:block"]}],'
			'"methods":[{"method_name":"m_do_put_on_already_satisfied","task_name":"do_put_on",'
			'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
			'"context":["on(?x, ?y)"],"subtasks":[],"ordering":[]}]}'
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].parameters == ("?x", "?y")


def test_library_postprocess_truncates_auxiliary_task_args_to_task_arity() -> None:
	synthesizer = HTNMethodSynthesizer()
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_clear","parameters":["?x:block"]}],'
		'"methods":[{"method_name":"m_do_clear_via_unstack","task_name":"do_clear",'
		'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
		'"context":["on(?y, ?x)","clear(?y)","handempty"],'
		'"subtasks":[{"step_id":"s1","task_name":"unstack","args":["?y","?x"],"kind":"primitive"}],'
		'"ordering":[]}]}'
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].task_args == ("?x",)


def test_library_postprocess_drops_self_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_clear","parameters":["?x:block"]}],'
		'"methods":[{"method_name":"m_do_clear_via_putdown","task_name":"do_clear",'
		'"parameters":["?x:block"],"task_args":["?x"],"context":["holding(?x)"],'
		'"subtasks":[{"step_id":"s1","task_name":"put_down","args":["?x"],"kind":"primitive"}],'
		'"ordering":[["s1","s1"],["s1","s1"]]}]}'
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].ordering == ()


def test_library_postprocess_collects_missing_variables_from_context_and_steps() -> None:
	synthesizer = HTNMethodSynthesizer()

	parameters = synthesizer._normalise_method_parameters(
		("?x",),
		context=(HTNLiteral(predicate="on", args=("?y", "?x")),),
		steps=(
			HTNMethodStep(
				step_id="s1",
				task_name="stack",
				args=("?y", "?z"),
				kind="primitive",
				preconditions=(),
				effects=(),
				literal=None,
				action_name=None,
			),
		),
	)

	assert parameters == ("?x", "?y", "?z")


def test_library_postprocess_drops_true_literals_from_method_context() -> None:
	synthesizer = HTNMethodSynthesizer()
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_move","parameters":["?x:block","?y:block"]}],'
		'"methods":[{"method_name":"m_do_move","task_name":"do_move",'
		'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
		'"context":["true","holding(?x)"],'
		'"subtasks":[{"step_id":"s1","task_name":"stack","args":["?x","?y"],"kind":"primitive"}],'
		'"ordering":[]}]}'
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert [literal.predicate for literal in normalised.methods[0].context] == ["holding"]


def test_parse_llm_library_accepts_object_pair_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_put_on","parameters":["?x:block","?y:block"]}],'
		'"methods":[{"method_name":"m_do_put_on_direct","task_name":"do_put_on",'
		'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
		'"context":["holding(?x)"],'
		'"subtasks":[{"step_id":"s1","task_name":"do_clear","args":["?y"],"kind":"compound"},'
		'{"step_id":"s2","task_name":"stack","args":["?x","?y"],"kind":"primitive"}],'
		'"ordering":[{"pre":"s1","post":"s2"}]}]}'
	)

	assert library.methods[0].ordering == (("s1", "s2"),)


def test_parse_llm_library_accepts_first_second_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"auto_calibrate","parameters":["?s:satellite","?i:instrument"]}],'
		'"methods":[{"method_name":"m_auto_calibrate","task_name":"auto_calibrate",'
		'"parameters":["?s:satellite","?i:instrument"],"task_args":["?s","?i"],'
		'"context":["on_board(?i, ?s)"],'
		'"subtasks":[{"step_id":"s1","task_name":"switch_on","args":["?i","?s"],"kind":"primitive"},'
		'{"step_id":"s2","task_name":"calibrate","args":["?s","?i","?d"],"kind":"primitive"}],'
		'"ordering":[{"first":"s1","second":"s2"}]}]}'
	)

	assert library.methods[0].ordering == (("s1", "s2"),)


def test_parse_llm_library_accepts_precedent_subsequent_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_on_table","parameters":["?x:block"]}],'
		'"methods":[{"method_name":"m_do_on_table_chain","task_name":"do_on_table",'
		'"parameters":["?x:block","?z:block"],"task_args":["?x"],'
		'"context":[],"subtasks":['
		'{"step_id":"s1","task_name":"unstack","args":["?x","?z"],"kind":"primitive"},'
		'{"step_id":"s2","task_name":"put-down","args":["?x"],"kind":"primitive"}],'
		'"ordering":[{"precedent":"s1","subsequent":"s2"}]}]}'
	)

	assert library.methods[0].ordering == (("s1", "s2"),)


def test_parse_llm_library_accepts_sup_sub_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"calibrate_abs","parameters":["?i:instrument"]}],'
		'"methods":[{"method_name":"m_calibrate_abs_chain","task_name":"calibrate_abs",'
		'"parameters":["?i:instrument","?s:satellite"],"task_args":["?i"],'
		'"context":[],"subtasks":['
		'{"step_id":"s1","task_name":"switch_on","args":["?i","?s"],"kind":"primitive"},'
		'{"step_id":"s2","task_name":"calibrate","args":["?s","?i","?d"],"kind":"primitive"}],'
		'"ordering":[{"sup":"s1","sub":"s2"}]}]}'
	)

	assert library.methods[0].ordering == (("s1", "s2"),)


def test_parse_llm_library_accepts_localized_ordering_edges() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_observation","parameters":["?d:image_direction","?m:mode"]}],'
		'"methods":[{"method_name":"m_do_observation_chain","task_name":"do_observation",'
		'"parameters":["?d:image_direction","?m:mode","?s:satellite","?i:instrument"],'
		'"task_args":["?d","?m"],"context":[],'
		'"subtasks":['
		'{"step_id":"s1","task_name":"switch_on","args":["?i","?s"],"kind":"primitive"},'
		'{"step_id":"s2","task_name":"take_image","args":["?s","?d","?i","?m"],"kind":"primitive"}],'
		'"ordering":[{"先行":"s1","后继":"s2"}]}]}'
	)

	assert library.methods[0].ordering == (("s1", "s2"),)


def test_parse_llm_library_splits_top_level_conjunction_context_strings() -> None:
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_move","parameters":["?x:block","?y:block"]}],'
		'"methods":[{"method_name":"m_do_move_direct","task_name":"do_move",'
		'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
		'"context":["clear(?x), ontable(?x), handempty"],'
		'"subtasks":[{"step_id":"s1","task_name":"pick-up","args":["?x"],"kind":"primitive"}],'
		'"ordering":[]}]}'
	)

	assert [literal.to_signature() for literal in library.methods[0].context] == [
		"clear(?x)",
		"ontable(?x)",
		"handempty",
	]


def test_library_postprocess_normalises_fused_negation_predicates_to_negative_literals() -> None:
	synthesizer = HTNMethodSynthesizer()
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = synthesizer._parse_llm_library(
		'{"compound_tasks":[{"name":"do_move","parameters":["?x:block","?y:block"]}],'
		'"methods":[{"method_name":"m_do_move_direct","task_name":"do_move",'
		'"parameters":["?x:block","?y:block"],"task_args":["?x","?y"],'
		'"context":["noton(?x,?y)","clear(?x)","ontable(?x)","handempty"],'
		'"subtasks":[{"step_id":"s1","task_name":"pick-up","args":["?x"],"kind":"primitive"}],'
		'"ordering":[]}]}'
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert [literal.to_signature() for literal in normalised.methods[0].context] == [
		"!on(?x, ?y)",
		"clear(?x)",
		"ontable(?x)",
		"handempty",
	]


def test_parse_llm_library_salvages_domain_task_payload_with_extra_closers() -> None:
	synthesizer = HTNMethodSynthesizer()
	response_text = (
		'{"tasks":['
		'{"name":"calibrate_abs","parameters":["R","C"],'
		'"constructive":[{"precondition":["calibrated(C)"],"ordered_subtasks":[]}]}}]},'
		'{"name":"empty_store","parameters":["S","R"],'
		'"constructive":[{"precondition":["empty(S)"],"ordered_subtasks":[]}]}}]}'
	)

	library = synthesizer._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["calibrate_abs", "empty_store"]


def test_parse_llm_library_repairs_common_stray_quote_before_field_separator() -> None:
	response_text = (
		'{"tasks":[{"name":"do_clear","parameters":["ARG1"],'
		'"constructive":[{"producer":"unstack(ARG2, ARG1)"}]","context":["clear(ARG1)"]}]}'
	)

	library = HTNMethodSynthesizer()._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["do_clear"]


def test_parse_llm_library_repairs_missing_string_quote_before_object_closer() -> None:
	response_text = (
		'{"tasks":['
		'{"name":"get_image_data","parameters":["?objective","?mode"],'
		'"constructive":[{"producer":"take_image(?rover, ?wp, ?objective, ?camera, ?mode)}]},'
		'{"name":"navigate_abs","parameters":["?rover","?to"],'
		'"constructive":[{"producer":"navigate(?rover, ?from, ?to)"}]}'
		']}'
	)

	library = HTNMethodSynthesizer()._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["get_image_data", "navigate_abs"]


def test_parse_llm_library_unwraps_single_list_wrapped_tasks_payload() -> None:
	response_text = (
		'[{"tasks":['
		'{"name":"auto_calibrate","parameters":["?s","?i"],'
		'"context":["calibrated(?i)"],'
		'"constructive":[{"producer":"calibrate(?s, ?i, ?d)"}]}'
		']}]'
	)

	library = HTNMethodSynthesizer()._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["auto_calibrate"]


def test_parse_llm_library_repairs_missing_task_object_opener_before_name_field() -> None:
	response_text = (
		'{"tasks":['
		'{"name":"deliver","parameters":["?p","?l"],'
		'"constructive":[{"producer":"drop(?v, ?l, ?p, ?s1, ?s2)"}]},"name":"get-to",'
		'"parameters":["?v","?l"],'
		'"constructive":[{"producer":"drive(?v, ?l1, ?l2)"}]}'
		']}'
	)

	library = HTNMethodSynthesizer()._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["deliver", "get-to"]


def test_plan_library_artifact_bundle_round_trips_masked_domain_and_asl_files(tmp_path: Path) -> None:
	method_library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	artifact = PlanLibraryArtifactBundle(
		domain_name="blocksworld",
		query_sequence=(),
		temporal_specifications=(),
		method_library=method_library,
		plan_library=PlanLibrary(domain_name="blocksworld", plans=()),
		translation_coverage=TranslationCoverage(
			domain_name="blocksworld",
			methods_considered=len(tuple(method_library.methods or ())),
			plans_generated=0,
			accepted_translation=0,
		),
		library_validation=LibraryValidationRecord(
			library_id="blocksworld",
			passed=True,
			method_count=len(tuple(method_library.methods or ())),
			plan_count=0,
			checked_layers={},
		),
		method_synthesis_metadata={"llm_attempted": True, "source_domain_kind": "masked_official"},
		artifact_root=str((tmp_path / "artifact").resolve()),
	)

	paths = persist_plan_library_artifact_bundle(
		artifact_root=tmp_path / "artifact",
		artifact=artifact,
		masked_domain_text="(define (domain masked-blocksworld))\n",
		plan_library_asl_text="+!start <- true.\n",
	)
	loaded = load_plan_library_artifact_bundle(tmp_path / "artifact")

	assert Path(paths["masked_domain"]).read_text() == "(define (domain masked-blocksworld))\n"
	assert Path(paths["plan_library_asl"]).read_text() == "+!start <- true.\n"
	assert loaded.domain_name == "blocksworld"
	assert loaded.method_synthesis_metadata["source_domain_kind"] == "masked_official"
	assert loaded.artifact_root == str((tmp_path / "artifact").resolve())
	assert loaded.masked_domain_file == str((tmp_path / "artifact" / "masked_domain.hddl").resolve())
	assert loaded.plan_library_asl_file == str((tmp_path / "artifact" / "plan_library.asl").resolve())


def test_build_domain_library_synthesizes_from_masked_domain_only(tmp_path: Path) -> None:
	pipeline = PlanLibraryGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])
	pipeline.logger = ExecutionLogger(logs_dir=str(tmp_path / "logs"), run_origin="tests")
	official_method_library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	captured: dict[str, object] = {}

	def fake_synthesise_domain_methods(
		*,
		synthesis_domain=None,
		source_domain_kind="official",
		masked_domain_file=None,
		original_method_count=None,
		**_kwargs,
	):
		assert synthesis_domain is not None
		assert len(list(getattr(synthesis_domain, "methods", ()))) == 0
		captured["source_domain_kind"] = source_domain_kind
		captured["masked_domain_file"] = masked_domain_file
		captured["original_method_count"] = original_method_count
		return official_method_library, {
			"used_llm": False,
			"llm_prompt": None,
			"llm_response": None,
			"llm_finish_reason": None,
			"llm_attempts": 0,
			"llm_generation_attempts": 0,
			"llm_response_time_seconds": None,
			"llm_attempt_durations_seconds": [],
			"prompt_strategy": "compact_domain_contracts",
			"prompt_declared_task_count": len(official_method_library.compound_tasks),
			"prompt_domain_task_contract_count": len(official_method_library.compound_tasks),
			"prompt_reusable_dynamic_resource_count": 0,
			"llm_request_count": 0,
			"domain_task_contracts": [],
			"action_analysis": {},
			"derived_analysis": {},
			"failure_class": None,
			"declared_compound_tasks": [task.name for task in official_method_library.compound_tasks],
			"compound_tasks": len(official_method_library.compound_tasks),
			"primitive_tasks": len(official_method_library.primitive_tasks),
			"methods": len(official_method_library.methods),
			"model": None,
		}

	orchestrator = pipeline._orchestrator
	orchestrator.synthesise_domain_methods = fake_synthesise_domain_methods  # type: ignore[method-assign]
	orchestrator.validate_method_library = lambda method_library, **kwargs: {  # type: ignore[method-assign]
		"validated_task_count": len(method_library.compound_tasks),
		"layers": {
			"signature_conformance": {"passed": True, "warnings": []},
			"typed_structural_soundness": {"passed": True, "warnings": []},
			"decomposition_admissibility": {"passed": True, "warnings": []},
			"materialized_parseability": {"passed": True, "warnings": []},
		},
	}

	result = pipeline.build_library_bundle(output_root=str(tmp_path / "artifact"))

	assert result["success"] is True
	assert captured["source_domain_kind"] == "masked_official"
	assert str(captured["masked_domain_file"]).endswith("masked_domain.hddl")
	assert int(captured["original_method_count"]) > 0
	assert Path(result["artifact_paths"]["masked_domain"]).exists()
	assert Path(result["artifact_paths"]["plan_library_asl"]).exists()


def test_plan_library_public_entrypoint_does_not_import_domain_complete_pipeline() -> None:
	pipeline_source = (PROJECT_ROOT / "src" / "plan_library" / "pipeline.py").read_text()

	assert "pipeline.domain_complete_pipeline" not in pipeline_source
	assert "DomainCompletePipeline" not in pipeline_source


def test_method_library_validation_reports_structural_admissibility_and_does_not_plan() -> None:
	class FakeContext:
		def __init__(self, domain):
			self.domain = domain
			self.output_dir = PROJECT_ROOT / "tests" / "generated" / "tmp_gate"
			self.domain_type_names = {"block", "object"}
			self.type_parent_map = {"block": "object", "object": None}
			self.logger = type(
				"FakeLogger",
				(),
				{
					"log_domain_gate": staticmethod(lambda *args, **kwargs: None),
				},
			)()

		@staticmethod
		def _sanitize_name(value: str) -> str:
			return str(value).strip().replace("-", "_")

		@staticmethod
		def _emit_domain_gate_progress(message: str) -> None:
			_ = message

		@staticmethod
		def _record_step_timing(step: str, stage_start: float, breakdown=None, metadata=None) -> None:
			_ = (step, stage_start, breakdown, metadata)

		def _task_type_signature(self, task_name, method_library):
			_ = method_library
			for task in getattr(self.domain, "tasks", ()):
				if getattr(task, "name", None) == task_name:
					return tuple(
						self._parse_parameter_type(parameter)
						for parameter in getattr(task, "parameters", ())
					)
			return ()

		@staticmethod
		def _parse_parameter_type(parameter: str) -> str:
			text = str(parameter or "")
			if ":" in text:
				return text.split(":", 1)[1].strip()
			return "object"

	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	method_library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	validator = MethodLibraryValidator(FakeContext(domain))

	summary = validator.validate(method_library)

	assert summary["gate_profile"] == "structural_admissibility"
	for layer_name in (
		"signature_conformance",
		"typed_structural_soundness",
		"decomposition_admissibility",
		"materialized_parseability",
	):
		layer = summary[layer_name]
		assert set(layer) == {"passed", "checked_count", "failure_reason", "warnings"}
		assert layer["passed"] is True
	assert summary["validated_task_count"] == len(method_library.compound_tasks)
	assert all(
		record["validation_mode"] == "structural_admissibility"
		for record in summary["task_validations"]
	)
	assert all("plan" not in record for record in summary["task_validations"])


def test_plan_library_generation_pipeline_uses_dedicated_generation_orchestrator() -> None:
	pipeline = PlanLibraryGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])

	assert pipeline._orchestrator.__class__.__name__ == "PlanLibraryGenerationOrchestrator"


def test_generation_pipeline_exposes_domain_only_context() -> None:
	pipeline = PlanLibraryGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])

	assert pipeline.context.domain_file == DOMAIN_FILES["blocksworld"]
	assert not hasattr(pipeline.context, "problem")
	assert not hasattr(pipeline.context, "problem_file")


def test_domain_complete_coverage_requires_executable_method_for_each_declared_task() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	library_without_move = HTNMethodLibrary(
		compound_tasks=[task for task in library.compound_tasks if task.name != "do_move"],
		primitive_tasks=list(library.primitive_tasks),
		methods=list(library.methods),
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="omitted declared compound tasks"):
		validate_domain_complete_coverage(domain, library_without_move)
