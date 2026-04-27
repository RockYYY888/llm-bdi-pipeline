from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from language_model import (
	OPENAI_COMPATIBLE_JSON_PROFILE_NAME,
	build_json_chat_completion_kwargs,
	create_openai_compatible_json_completion,
)


def test_json_chat_completion_kwargs_use_one_openai_compatible_shape() -> None:
	payload = build_json_chat_completion_kwargs(
		model="provider/model",
		messages=[{"role": "system", "content": "Return JSON."}],
		timeout=12.5,
		max_tokens=128,
		stream=False,
		reasoning_effort="high",
		thinking_type="enabled",
	)

	assert OPENAI_COMPATIBLE_JSON_PROFILE_NAME == "openai_compatible_json_chat"
	assert payload == {
		"model": "provider/model",
		"messages": [{"role": "system", "content": "Return JSON."}],
		"temperature": 0.0,
		"stream": False,
		"response_format": {"type": "json_object"},
		"timeout": 12.5,
		"max_tokens": 128,
		"reasoning_effort": "high",
		"extra_body": {"thinking": {"type": "enabled"}},
	}


def test_json_chat_completion_helper_delegates_to_client() -> None:
	captured_kwargs: dict[str, object] = {}

	class FakeCompletions:
		def create(self, **kwargs):
			captured_kwargs.update(kwargs)
			return {"ok": True}

	class FakeChat:
		def __init__(self) -> None:
			self.completions = FakeCompletions()

	class FakeClient:
		def __init__(self) -> None:
			self.chat = FakeChat()

	response = create_openai_compatible_json_completion(
		FakeClient(),
		model="provider/model",
		messages=[{"role": "user", "content": "Generate JSON."}],
		max_tokens=64,
	)

	assert response == {"ok": True}
	assert captured_kwargs["model"] == "provider/model"
	assert captured_kwargs["response_format"] == {"type": "json_object"}
	assert captured_kwargs["max_tokens"] == 64
