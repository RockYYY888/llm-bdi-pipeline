from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from offline_method_generation.method_synthesis.domain_materialization import write_masked_domain_file
from offline_method_generation.method_synthesis.errors import HTNSynthesisError, LLMStreamingResponseError
from offline_method_generation.domain_gate.validator import OfflineDomainGateValidator
from offline_method_generation.method_synthesis.minimal_validation import validate_domain_complete_coverage
from offline_method_generation.method_synthesis.prompts import (
	build_domain_htn_system_prompt,
	build_domain_prompt_analysis_payload,
	build_domain_htn_user_prompt,
)
from offline_method_generation.method_synthesis.domain_prompts import _render_method_blueprint_blocks
from offline_method_generation.method_synthesis.schema import HTNLiteral, HTNMethodLibrary, HTNMethodStep
from offline_method_generation.method_synthesis.synthesizer import HTNMethodSynthesizer
from offline_method_generation.pipeline import OfflineMethodGenerationPipeline
from offline_method_generation.artifacts import DomainLibraryArtifact, load_domain_library_artifact, persist_domain_library_artifact
from pipeline.execution_logger import ExecutionLogger
from tests.support.offline_generation_support import (
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


def test_domain_prompt_is_query_independent_and_does_not_leak_official_methods(tmp_path: Path) -> None:
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
	)

	for official_method in official_domain.methods:
		assert official_method.name not in system_prompt
		assert official_method.name not in user_prompt

	assert "problem.hddl" not in user_prompt
	assert "target_literals" not in user_prompt
	assert "target_task_bindings" not in user_prompt
	assert "Do not condition the library on any benchmark query." in user_prompt
	assert "primitive_action_schemas:" in user_prompt
	assert "method_blueprints" in user_prompt
	assert '"method_family_schemas"' in user_prompt
	assert '"uncovered_prerequisite_families"' in user_prompt
	assert "never reuse one AUX or task variable across argument positions whose declared types are incompatible" in user_prompt
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
	assert "acquire" in rendered["uncovered_prerequisite_families"][0]
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


def test_library_postprocess_bound_helpers_are_callable_via_synthesizer_instance() -> None:
	synthesizer = HTNMethodSynthesizer()

	assert synthesizer._domain_schema_hint().startswith('{"compound_tasks"')
	assert synthesizer._looks_like_variable("ARG1")
	assert synthesizer._looks_like_variable("?x")


def test_method_synthesis_transport_streams_on_openrouter_base_url() -> None:
	synthesizer = HTNMethodSynthesizer(
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
	)

	assert synthesizer._should_stream_method_synthesis_response()
	assert synthesizer._method_synthesis_request_plugins(True) is None


def test_method_synthesis_transport_omits_response_format_on_openrouter_streaming() -> None:
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
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
	)
	synthesizer.client = FakeClient()

	synthesizer._create_chat_completion({"system": "x", "user": "y"}, max_tokens=128)

	assert captured_kwargs["stream"] is True
	assert "response_format" not in captured_kwargs


def test_method_synthesis_transport_omits_kimi_completion_ceiling() -> None:
	synthesizer = HTNMethodSynthesizer(model="moonshotai/kimi-k2.6")

	assert synthesizer._apply_method_synthesis_provider_token_ceiling(48000) is None


def test_method_synthesis_request_profile_uses_single_pass_context_budget() -> None:
	synthesizer = HTNMethodSynthesizer(
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
	)
	prompt = {
		"system": "abcd",
		"user": "efgh",
	}

	profile = synthesizer._method_synthesis_request_profile(prompt=prompt)
	expected_prompt_tokens = synthesizer._estimate_method_synthesis_prompt_token_budget(prompt)

	assert profile["name"] == "kimi_stream_single_pass"
	assert profile["context_window_tokens"] == 196608
	assert profile["prompt_token_estimate"] == expected_prompt_tokens
	assert profile["answer_token_reserve"] == 5805
	assert profile["context_margin_tokens"] == 19661
	assert profile["reasoning_headroom_tokens"] == (
		196608 - 19661 - 5805 - 2048 - expected_prompt_tokens
	)
	assert profile["reasoning_headroom_ratio"] == 0.70
	assert profile["transport_overhead_tokens"] == 2048
	assert profile["reasoning_max_tokens"] == int(
		(196608 - 19661 - 5805 - 2048 - expected_prompt_tokens) * 0.70,
	)
	assert profile["first_chunk_timeout_seconds"] == 300.0
	assert profile["session_id"] == "offline-method-generation"


def test_method_synthesis_transport_omits_reasoning_field_when_budget_is_zero() -> None:
	synthesizer = HTNMethodSynthesizer(
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
	)

	extra_body = synthesizer._openrouter_provider_routing_body(
		request_profile={
			"name": "kimi_stream_single_pass",
			"stream_response": True,
			"reasoning_max_tokens": 0,
			"first_chunk_timeout_seconds": 300.0,
		},
	)

	assert extra_body == {
		"provider": {
			"only": ["minimax"],
			"allow_fallbacks": False,
		},
		"session_id": "offline-method-generation",
	}


def test_method_synthesis_transport_keeps_response_format_off_openrouter_streaming_path() -> None:
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

	synthesizer = HTNMethodSynthesizer(model="other/model", base_url="https://api.example.com/v1")
	synthesizer.client = FakeClient()

	synthesizer._create_chat_completion({"system": "x", "user": "y"}, max_tokens=128)

	assert captured_kwargs["stream"] is False
	assert captured_kwargs["response_format"] == {"type": "json_object"}


def test_method_synthesis_transport_uses_raw_openrouter_streaming_path() -> None:
	captured_request = {}

	class FakeSynthesizer(HTNMethodSynthesizer):
		def _create_raw_openrouter_stream_response(
			self,
			request_kwargs,
			*,
			request_timeout_seconds=None,
		):
			captured_request["request_kwargs"] = dict(request_kwargs)
			captured_request["request_timeout_seconds"] = request_timeout_seconds
			return object()

	synthesizer = FakeSynthesizer(
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
		api_key="sk-test",
		timeout=60,
	)

	response = synthesizer._create_chat_completion(
		{"system": "x", "user": "y"},
		max_tokens=16,
		request_profile={
			"name": "kimi_stream_single_pass",
			"stream_response": True,
			"reasoning_max_tokens": 8,
			"first_chunk_timeout_seconds": 90.0,
		},
		request_timeout_seconds=90.0,
	)

	assert response is not None
	assert captured_request["request_kwargs"]["stream"] is True
	assert "response_format" not in captured_request["request_kwargs"]
	assert captured_request["request_timeout_seconds"] == 90.0
	assert captured_request["request_kwargs"]["extra_body"]["session_id"] == "offline-method-generation"


def test_method_synthesis_transport_omits_max_tokens_for_kimi_requests() -> None:
	captured_request = {}

	class FakeSynthesizer(HTNMethodSynthesizer):
		def _create_raw_openrouter_stream_response(
			self,
			request_kwargs,
			*,
			request_timeout_seconds=None,
		):
			captured_request["request_kwargs"] = dict(request_kwargs)
			captured_request["request_timeout_seconds"] = request_timeout_seconds
			return object()

	synthesizer = FakeSynthesizer(
		model="moonshotai/kimi-k2.6",
		base_url="https://openrouter.ai/api/v1",
		api_key="sk-test",
		timeout=60,
	)

	synthesizer._create_chat_completion(
		{"system": "x", "user": "y"},
		max_tokens=None,
		request_profile={
			"name": "kimi_stream_single_pass",
			"stream_response": True,
			"reasoning_max_tokens": 0,
			"first_chunk_timeout_seconds": 300.0,
		},
		request_timeout_seconds=300.0,
	)

	assert "max_tokens" not in captured_request["request_kwargs"]
	assert captured_request["request_kwargs"]["extra_body"]["session_id"] == "offline-method-generation"


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
				"name": "kimi_stream_single_pass",
				"stream_response": True,
				"reasoning_max_tokens": 8,
				"first_chunk_timeout_seconds": 0.01,
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
				transport_metadata["llm_response_mode"] = "streaming"
				transport_metadata["llm_request_profile"] = dict(request_profile or {}).get("name")
			time.sleep(0.05)
			return "{}", "stop", dict(transport_metadata or {})

	synthesizer = SlowSynthesizer(timeout=0.01)

	with pytest.raises(TimeoutError, match="first-chunk deadline|configured timeout") as exc_info:
		synthesizer._call_llm({"system": "x", "user": "y"}, max_tokens=16)

	assert getattr(exc_info.value, "transport_metadata", {}) == {
		"llm_request_id": "req_timeout",
		"llm_response_mode": "streaming",
		"llm_request_profile": "kimi_stream_single_pass",
		"llm_reasoning_budget": 123526,
		"llm_first_chunk_timeout_seconds": 300.0,
		"llm_context_window_tokens": 204800,
		"llm_prompt_token_estimate": 1,
		"llm_answer_token_reserve": 5805,
		"llm_context_margin_tokens": 20480,
		"llm_reasoning_headroom_tokens": 176466,
		"llm_reasoning_headroom_ratio": 0.70,
		"llm_transport_overhead_tokens": 2048,
		"llm_session_id": "offline-method-generation",
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
	assert transport_metadata["llm_complete_json_seconds"] >= 0.0


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

def test_method_synthesis_transport_counts_reasoning_as_stream_activity() -> None:
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
	assert transport_metadata["llm_first_reasoning_chunk_seconds"] >= 0.0
	assert "llm_first_content_chunk_seconds" not in transport_metadata


def test_method_synthesis_fails_immediately_on_stream_failure() -> None:
	class SinglePassSynthesizer(HTNMethodSynthesizer):
		def __init__(self):
			super().__init__(model="moonshotai/kimi-k2.6")
			self.call_count = 0

		def _call_llm(self, prompt, *, max_tokens=None):
			self.call_count += 1
			error = LLMStreamingResponseError(
				"LLM response did not contain usable textual JSON content. finish_reason='length'",
				finish_reason="length",
			)
			error.transport_metadata = {
				"llm_request_id": "req_retry_1",
				"llm_response_mode": "streaming",
				"llm_request_profile": "kimi_stream_single_pass",
			}
			raise error

	synthesizer = SinglePassSynthesizer()
	metadata = {}
	with pytest.raises(HTNSynthesisError, match="LLM request failed"):
		synthesizer._request_complete_llm_library(
			{"system": "x", "user": "y"},
			domain=type("FakeDomain", (), {"actions": [], "tasks": [], "predicates": []})(),
			metadata=metadata,
			max_tokens=256,
		)

	assert synthesizer.call_count == 1
	assert metadata["llm_attempts"] == 1
	assert metadata["llm_attempt_trace"][0]["request_id"] == "req_retry_1"
	assert metadata["llm_attempt_trace"][0]["request_profile"] == "kimi_stream_single_pass"


def test_method_synthesis_transport_preserves_reasoning_preview_when_no_json_arrives() -> None:
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
				reasoning="Now for all noop methods, parameters are the task signature parameters.",
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
	assert transport_metadata["llm_reasoning_characters"] > 0
	assert "top-level keys compound_tasks and methods" in transport_metadata["llm_reasoning_preview"]


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
		'"methods":[{"method_name":"m_do_put_on_noop","task_name":"do_put_on",'
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
		'"noop":{"precondition":["calibrated(C)"]}}]},'
		'{"name":"empty_store","parameters":["S","R"],'
		'"noop":{"precondition":["empty(S)"]}}]}'
	)

	library = synthesizer._parse_llm_library(response_text)

	assert [task.name for task in library.compound_tasks] == ["calibrate_abs", "empty_store"]


def test_parse_llm_library_repairs_common_stray_quote_before_field_separator() -> None:
	response_text = (
		'{"tasks":[{"name":"do_clear","parameters":["ARG1"],'
		'"constructive":[{"producer":"unstack(ARG2, ARG1)"}]","noop":{"precondition":["clear(ARG1)"]}}]}'
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
		'"noop":{"precondition":["calibrated(?i)"]},'
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


def test_domain_library_artifact_round_trips_masked_and_generated_domain_files(tmp_path: Path) -> None:
	method_library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	artifact = DomainLibraryArtifact(
		domain_name="blocksworld",
		method_library=method_library,
		method_synthesis_metadata={"llm_attempted": True, "source_domain_kind": "masked_official"},
		domain_gate={"validated_task_count": 4},
		source_domain_kind="masked_official",
	)

	paths = persist_domain_library_artifact(
		artifact_root=tmp_path / "artifact",
		artifact=artifact,
		masked_domain_text="(define (domain masked-blocksworld))\n",
		generated_domain_text="(define (domain generated-blocksworld))\n",
	)
	loaded = load_domain_library_artifact(tmp_path / "artifact")

	assert Path(paths["masked_domain"]).read_text() == "(define (domain masked-blocksworld))\n"
	assert Path(paths["generated_domain"]).read_text() == "(define (domain generated-blocksworld))\n"
	assert loaded.domain_name == "blocksworld"
	assert loaded.source_domain_kind == "masked_official"
	assert loaded.method_synthesis_metadata["source_domain_kind"] == "masked_official"
	assert loaded.artifact_root == str((tmp_path / "artifact").resolve())
	assert loaded.masked_domain_file == str((tmp_path / "artifact" / "masked_domain.hddl").resolve())
	assert loaded.generated_domain_file == str((tmp_path / "artifact" / "generated_domain.hddl").resolve())


def test_build_domain_library_synthesizes_from_masked_domain_only(tmp_path: Path) -> None:
	pipeline = OfflineMethodGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])
	pipeline.logger = ExecutionLogger(logs_dir=str(tmp_path / "logs"), run_origin="tests")
	official_method_library = build_method_library_from_domain_file(DOMAIN_FILES["blocksworld"])
	captured: dict[str, object] = {}

	def fake_synthesise_domain_methods(
		*,
		synthesis_domain=None,
		source_domain_kind="official",
		masked_domain_file=None,
		original_method_count=None,
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
	orchestrator.validate_domain_library = lambda method_library: {  # type: ignore[method-assign]
		"validated_task_count": len(method_library.compound_tasks),
	}

	result = pipeline.build_domain_library(output_root=str(tmp_path / "artifact"))

	assert result["success"] is True
	assert captured["source_domain_kind"] == "masked_official"
	assert str(captured["masked_domain_file"]).endswith("masked_domain.hddl")
	assert int(captured["original_method_count"]) > 0
	assert Path(result["artifact_paths"]["masked_domain"]).exists()
	assert Path(result["artifact_paths"]["generated_domain"]).exists()


def test_offline_public_entrypoint_does_not_import_domain_complete_pipeline() -> None:
	pipeline_source = (PROJECT_ROOT / "src" / "offline_method_generation" / "pipeline.py").read_text()

	assert "pipeline.domain_complete_pipeline" not in pipeline_source
	assert "DomainCompletePipeline" not in pipeline_source


def test_offline_domain_gate_is_legality_only_and_does_not_plan() -> None:
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
	validator = OfflineDomainGateValidator(FakeContext(domain))

	summary = validator.validate(method_library)

	assert summary["gate_profile"] == "legality_only"
	assert summary["validated_task_count"] == len(method_library.compound_tasks)
	assert all(record["validation_mode"] == "legality_only" for record in summary["task_validations"])
	assert all("plan" not in record for record in summary["task_validations"])


def test_offline_method_generation_pipeline_uses_dedicated_offline_orchestrator() -> None:
	pipeline = OfflineMethodGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])

	class FakeOrchestrator:
		def __init__(self) -> None:
			self.calls: list[object] = []

		def build_domain_library(self, *, output_root=None):
			self.calls.append(output_root)
			return {"success": True, "output_root": output_root}

	fake_orchestrator = FakeOrchestrator()
	pipeline._orchestrator = fake_orchestrator  # type: ignore[assignment]

	result = pipeline.build_domain_library(output_root="/tmp/generated-domain")

	assert result == {"success": True, "output_root": "/tmp/generated-domain"}
	assert fake_orchestrator.calls == ["/tmp/generated-domain"]


def test_offline_pipeline_exposes_offline_only_context() -> None:
	pipeline = OfflineMethodGenerationPipeline(domain_file=DOMAIN_FILES["blocksworld"])

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
