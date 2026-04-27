"""OpenAI-compatible JSON chat completion transport.

All live language-model calls in this project should pass through this module so
that provider details remain isolated from generation, grounding, and evaluation
logic.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

OPENAI_COMPATIBLE_JSON_PROFILE_NAME = "openai_compatible_json_chat"


def create_openai_compatible_client(
	*,
	api_key: str,
	base_url: Optional[str] = None,
	timeout: Optional[float] = None,
	max_retries: int = 0,
) -> Any:
	"""Create an OpenAI-compatible client with the project-wide retry policy."""

	from openai import OpenAI

	client_kwargs: dict[str, Any] = {
		"api_key": api_key,
		"max_retries": max(0, int(max_retries)),
	}
	if base_url:
		client_kwargs["base_url"] = base_url
	if timeout is not None:
		client_kwargs["timeout"] = float(timeout)
	return OpenAI(**client_kwargs)


def build_json_chat_completion_kwargs(
	*,
	model: str,
	messages: Sequence[Mapping[str, str]],
	timeout: Optional[float] = None,
	max_tokens: Optional[int] = None,
	stream: bool = False,
	temperature: float = 0.0,
	reasoning_effort: Optional[str] = None,
	thinking_type: Optional[str] = None,
) -> dict[str, Any]:
	"""Build one provider-neutral JSON chat completion request payload."""

	request_kwargs: dict[str, Any] = {
		"model": model,
		"messages": [dict(message) for message in messages],
		"temperature": float(temperature),
		"stream": bool(stream),
		"response_format": {"type": "json_object"},
	}
	if timeout is not None:
		request_kwargs["timeout"] = float(timeout)
	if max_tokens is not None:
		request_kwargs["max_tokens"] = max(int(max_tokens), 1)
	if reasoning_effort:
		request_kwargs["reasoning_effort"] = str(reasoning_effort)
	if thinking_type:
		request_kwargs["extra_body"] = {"thinking": {"type": str(thinking_type)}}
	return request_kwargs


def create_openai_compatible_json_completion(
	client: Any,
	*,
	model: str,
	messages: Sequence[Mapping[str, str]],
	timeout: Optional[float] = None,
	max_tokens: Optional[int] = None,
	stream: bool = False,
	temperature: float = 0.0,
	reasoning_effort: Optional[str] = None,
	thinking_type: Optional[str] = None,
) -> Any:
	"""Issue one OpenAI-compatible JSON chat completion request."""

	request_kwargs = build_json_chat_completion_kwargs(
		model=model,
		messages=messages,
		timeout=timeout,
		max_tokens=max_tokens,
		stream=stream,
		temperature=temperature,
		reasoning_effort=reasoning_effort,
		thinking_type=thinking_type,
	)
	return client.chat.completions.create(**request_kwargs)
