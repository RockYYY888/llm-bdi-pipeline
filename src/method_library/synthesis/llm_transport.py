"""Language-model transport and response extraction for method synthesis."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from method_library.synthesis.schema import HTNMethodLibrary
from .errors import LLMStreamingResponseError

KIMI_OPENROUTER_CONTEXT_WINDOW_TOKENS = 262_144
KIMI_DIRECT_CONTEXT_WINDOW_TOKENS = 204_800
KIMI_COMPLETION_MAX_TOKENS = 65_536
KIMI_REASONING_MAX_TOKENS = 8_192
KIMI_PROMPT_ESTIMATE_CHARS_PER_TOKEN = 2.0
KIMI_SINGLE_PASS_FIRST_CHUNK_TIMEOUT_SECONDS = 1000.0
METHOD_LIBRARY_GENERATION_SESSION_ID = "method-library-generation"


def _namespace_from_json_like(value: object) -> object:
	if isinstance(value, dict):
		return SimpleNamespace(
			**{
				key: _namespace_from_json_like(item)
				for key, item in value.items()
			},
		)
	if isinstance(value, list):
		return [_namespace_from_json_like(item) for item in value]
	return value


class _RawOpenRouterStreamingResponse:
	def __init__(
		self,
		*,
		client: Any,
		response: Any,
		handshake_seconds: Optional[float] = None,
	) -> None:
		self._client = client
		self._response = response
		self.handshake_seconds = handshake_seconds
		self.id = (
			response.headers.get("x-request-id")
			or response.headers.get("request-id")
			or None
		)

	def __iter__(self):
		for raw_line in self._response.iter_lines():
			if raw_line is None:
				continue
			line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
			text = line.strip()
			if not text.startswith("data:"):
				continue
			payload = text[5:].strip()
			if not payload:
				continue
			if payload == "[DONE]":
				break
			try:
				chunk = json.loads(payload)
			except json.JSONDecodeError:
				continue
			namespace_chunk = _namespace_from_json_like(chunk)
			if getattr(namespace_chunk, "id", None) is None and self.id:
				setattr(namespace_chunk, "id", self.id)
			yield namespace_chunk

	def close(self) -> None:
		try:
			self._response.close()
		finally:
			self._client.close()


class MethodSynthesisLLMTransportMixin:
	@staticmethod
	def _run_with_wall_clock_timeout(
		timeout_seconds: Optional[float],
		callback,
	):
		effective_timeout_seconds = float(timeout_seconds or 0.0)
		if effective_timeout_seconds <= 0.0:
			return callback()
		if threading.current_thread() is not threading.main_thread():
			return callback()
		if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
			return callback()

		def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
			_ = (signum, frame)
			raise TimeoutError(
				"Method-synthesis LLM request exceeded the configured wall-clock "
				"timeout before a response object was created.",
			)

		previous_handler = signal.getsignal(signal.SIGALRM)
		try:
			signal.signal(signal.SIGALRM, _timeout_handler)
			signal.setitimer(signal.ITIMER_REAL, effective_timeout_seconds)
			return callback()
		finally:
			signal.setitimer(signal.ITIMER_REAL, 0.0)
			signal.signal(signal.SIGALRM, previous_handler)

	@staticmethod
	def _emit_method_synthesis_progress(message: str) -> None:
		if not str(os.getenv("METHOD_SYNTHESIS_PROGRESS", "")).strip():
			return
		sys.stderr.write(f"[METHOD SYNTHESIS PROGRESS] {message}\n")
		sys.stderr.flush()

	@staticmethod
	def _estimate_method_synthesis_response_token_budget(
		*,
		prompt_analysis: Optional[Dict[str, Any]],
		ast_compiler_defaults: Optional[Dict[str, Any]],
		default_max_tokens: int | None = None,
	) -> int:
		max_tokens = int(default_max_tokens or 0)
		if max_tokens <= 0:
			max_tokens = 8000
		defaults = dict(ast_compiler_defaults or {})
		task_count = len(dict(defaults.get("task_defaults") or {}))
		prompt_payload = dict(prompt_analysis or {})
		if task_count <= 0:
			task_count = len(
				list(prompt_payload.get("domain_task_contracts") or ())
				or list(prompt_payload.get("declared_compound_tasks") or ())
				or list(prompt_payload.get("query_task_contracts") or ())
				or list(prompt_payload.get("support_task_contracts") or ())
			)
		contract_payloads = list(prompt_payload.get("domain_task_contracts") or ())
		if not contract_payloads:
			contract_payloads = list(prompt_payload.get("query_task_contracts") or ())
		if not contract_payloads:
			contract_payloads = list(prompt_payload.get("support_task_contracts") or ())
		producer_template_count = sum(
			len(list(payload.get("producer_consumer_templates") or ()))
			for payload in contract_payloads
			if isinstance(payload, dict)
		)
		support_task_count = sum(
			len(list(payload.get("composition_support_tasks") or ()))
			for payload in contract_payloads
			if isinstance(payload, dict)
		)
		shared_prerequisite_count = sum(
			len(list(payload.get("shared_dynamic_prerequisites") or ()))
			for payload in contract_payloads
			if isinstance(payload, dict)
		)
		recursive_support_count = sum(
			len(list(payload.get("recursive_support_predicates") or ()))
			for payload in contract_payloads
			if isinstance(payload, dict)
		)
		if ast_compiler_defaults:
			# One-shot domain synthesis only needs enough headroom for the emitted JSON
			# library plus a very small amount of provider-side hidden reasoning.
			estimated = (
				2200
				+ 300 * task_count
				+ 80 * producer_template_count
				+ 40 * support_task_count
				+ 40 * shared_prerequisite_count
				+ 60 * recursive_support_count
			)
		else:
			estimated = 2200 + 900 * task_count
		minimum_budget = 3000
		return min(max_tokens, max(minimum_budget, estimated))

	def _apply_method_synthesis_provider_token_ceiling(
		self,
		requested_max_tokens: int | None,
	) -> int | None:
		"""
		Keep provider-specific handling explicit even when no extra cap is applied.

		For Kimi method synthesis, use an explicit output ceiling and a smaller
		reasoning ceiling so final JSON has enough budget while reasoning remains
		stream-visible as heartbeat events.
		"""
		model_name = str(self.model or "").strip().lower()
		if model_name.startswith("moonshotai/"):
			return max(int(requested_max_tokens or KIMI_COMPLETION_MAX_TOKENS), 1)
		requested = max(int(requested_max_tokens or 0), 1)
		return requested

	def _call_llm(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
	) -> Tuple[str, Optional[str], Dict[str, Any]]:
		timeout_seconds = float(self.timeout or 0.0)
		request_profile = self._method_synthesis_request_profile(prompt=prompt)
		transport_metadata: Dict[str, Any] = {
			"llm_request_profile": request_profile["name"],
			"llm_reasoning_budget": request_profile.get("reasoning_max_tokens"),
			"llm_first_chunk_timeout_seconds": request_profile.get("first_chunk_timeout_seconds"),
		}
		for metadata_key, profile_key in (
			("llm_context_window_tokens", "context_window_tokens"),
			("llm_prompt_token_estimate", "prompt_token_estimate"),
			("llm_completion_max_tokens", "completion_max_tokens"),
			("llm_answer_token_reserve", "answer_token_reserve"),
			("llm_context_margin_tokens", "context_margin_tokens"),
			("llm_reasoning_headroom_tokens", "reasoning_headroom_tokens"),
			("llm_reasoning_headroom_ratio", "reasoning_headroom_ratio"),
			("llm_reasoning_excluded", "reasoning_excluded"),
			("llm_transport_overhead_tokens", "transport_overhead_tokens"),
			("llm_session_id", "session_id"),
			("llm_max_tokens_policy", "max_tokens_policy"),
		):
			if request_profile.get(profile_key) is not None:
				transport_metadata[metadata_key] = request_profile.get(profile_key)
		request_timeout_seconds = self._method_synthesis_request_timeout_seconds(
			timeout_seconds=timeout_seconds,
			request_profile=request_profile,
		)
		if request_timeout_seconds > 0.0:
			transport_metadata["llm_request_timeout_seconds"] = request_timeout_seconds
		call_start = time.monotonic()
		if timeout_seconds <= 0:
			return self._call_llm_direct(
				prompt,
				max_tokens=max_tokens,
				transport_metadata=transport_metadata,
				request_profile=request_profile,
				request_timeout_seconds=request_timeout_seconds or None,
			)
		try:
			response_text, finish_reason, response_metadata = self._call_llm_direct(
				prompt,
				max_tokens=max_tokens,
				transport_metadata=transport_metadata,
				request_profile=request_profile,
				request_timeout_seconds=request_timeout_seconds or None,
			)
			elapsed_seconds = time.monotonic() - call_start
			if timeout_seconds > 0.0 and elapsed_seconds >= timeout_seconds:
				timeout_error = TimeoutError(
					"Method-synthesis LLM call exceeded the configured timeout before "
					"returning a usable response.",
				)
				try:
					setattr(timeout_error, "transport_metadata", dict(response_metadata))
				except Exception:
					pass
				raise timeout_error
			return response_text, finish_reason, response_metadata
		except Exception as exc:
			merged_metadata = dict(transport_metadata)
			merged_metadata.update(dict(getattr(exc, "transport_metadata", None) or {}))
			if self._looks_like_transport_timeout(exc):
				if (
					float(request_profile.get("first_chunk_timeout_seconds") or 0.0) > 0.0
					and merged_metadata.get("llm_first_chunk_seconds") is None
				):
					timeout_error = TimeoutError(
						"Method-synthesis LLM call exceeded the configured first-chunk "
						"deadline before any streaming content arrived.",
					)
				else:
					timeout_error = TimeoutError(
						"Method-synthesis LLM call exceeded the configured timeout before "
						"returning a usable response.",
					)
				try:
					setattr(timeout_error, "transport_metadata", merged_metadata)
				except Exception:
					pass
				raise timeout_error from exc
			try:
				setattr(exc, "transport_metadata", merged_metadata)
			except Exception:
				pass
			raise

	@staticmethod
	def _method_synthesis_request_timeout_seconds(
		*,
		timeout_seconds: float,
		request_profile: Dict[str, Any],
	) -> float:
		effective_timeout_seconds = float(timeout_seconds or 0.0)
		first_chunk_timeout_seconds = float(
			request_profile.get("first_chunk_timeout_seconds") or 0.0,
		)
		if first_chunk_timeout_seconds <= 0.0:
			return effective_timeout_seconds
		if effective_timeout_seconds <= 0.0:
			return first_chunk_timeout_seconds
		return min(effective_timeout_seconds, first_chunk_timeout_seconds)

	@staticmethod
	def _looks_like_transport_timeout(exc: BaseException) -> bool:
		exc_type_name = exc.__class__.__name__.lower()
		if "timeout" in exc_type_name:
			return True
		return "timeout" in str(exc).lower()

	def _call_llm_direct(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
		transport_metadata: Optional[Dict[str, Any]] = None,
		request_profile: Optional[Dict[str, Any]] = None,
		request_timeout_seconds: Optional[float] = None,
	) -> Tuple[str, Optional[str], Dict[str, Any]]:
		metadata = transport_metadata if transport_metadata is not None else {}
		response = self._create_chat_completion(
			prompt,
			max_tokens=max_tokens,
			request_profile=request_profile,
			request_timeout_seconds=request_timeout_seconds,
		)
		return self._consume_llm_response(
			response,
			transport_metadata=metadata,
			total_timeout_seconds=float(self.timeout or 0.0),
		)

	def _consume_llm_response(
		self,
		response: object,
		*,
		transport_metadata: Optional[Dict[str, Any]] = None,
		total_timeout_seconds: float = 0.0,
	) -> Tuple[str, Optional[str], Dict[str, Any]]:
		metadata = transport_metadata if transport_metadata is not None else {}
		if hasattr(response, "choices"):
			choice = response.choices[0]
			finish_reason = getattr(choice, "finish_reason", None)
			content = self._extract_response_text(response)
			metadata["llm_response_mode"] = "non_streaming"
			request_id = self._extract_transport_request_id(response)
			if request_id:
				metadata["llm_request_id"] = request_id
			return content, finish_reason, dict(metadata)
		return self._consume_streaming_llm_response(
			response,
			transport_metadata=metadata,
			total_timeout_seconds=total_timeout_seconds,
		)

	def _create_chat_completion(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
		request_profile: Optional[Dict[str, Any]] = None,
		request_timeout_seconds: Optional[float] = None,
	):
		profile = dict(request_profile or {})
		stream_response = bool(
			profile.get("stream_response")
			if "stream_response" in profile
			else self._should_stream_method_synthesis_response()
		)
		request_kwargs: Dict[str, Any] = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": prompt["system"]},
				{"role": "user", "content": prompt["user"]},
			],
			"temperature": 0.0,
			"timeout": request_timeout_seconds if request_timeout_seconds is not None else self.timeout,
			"stream": stream_response,
		}
		if max_tokens is not None:
			request_kwargs["max_tokens"] = max_tokens
		if not stream_response or "openrouter.ai" not in str(self.base_url or "").strip().lower():
			request_kwargs["response_format"] = {"type": "json_object"}
		extra_body = self._openrouter_provider_routing_body(request_profile=profile)
		plugins = self._method_synthesis_request_plugins(
			stream_response,
			request_profile=profile,
		)
		if plugins is not None:
			extra_body = dict(extra_body or {})
			extra_body["plugins"] = plugins
		if extra_body is not None:
			request_kwargs["extra_body"] = extra_body
		if self._should_use_raw_openrouter_stream_transport(profile):
			return self._create_raw_openrouter_stream_response(
				request_kwargs,
				request_timeout_seconds=request_timeout_seconds,
			)
		return self._run_with_wall_clock_timeout(
			request_timeout_seconds,
			lambda: self.client.chat.completions.create(
				**request_kwargs,
			),
		)

	def _should_use_raw_openrouter_stream_transport(
		self,
		request_profile: Dict[str, Any],
	) -> bool:
		base_url = str(self.base_url or "").strip().lower()
		return bool(request_profile.get("stream_response")) and "openrouter.ai" in base_url

	def _create_raw_openrouter_stream_response(
		self,
		request_kwargs: Dict[str, Any],
		*,
		request_timeout_seconds: Optional[float],
	):
		import httpx

		if not self.api_key:
			raise RuntimeError("OpenRouter streaming transport requires an API key.")
		base_url = str(self.base_url or "").rstrip("/")
		url = f"{base_url}/chat/completions"
		payload = {
			key: value
			for key, value in request_kwargs.items()
			if key not in {"timeout", "response_format", "extra_body"}
		}
		extra_body = dict(request_kwargs.get("extra_body") or {})
		payload.update(extra_body)
		timeout_value = request_timeout_seconds if request_timeout_seconds is not None else self.timeout
		client = httpx.Client(
			timeout=httpx.Timeout(timeout_value),
		)
		try:
			request = client.build_request(
				"POST",
				url,
				headers={
					"Authorization": f"Bearer {self.api_key}",
					"Content-Type": "application/json",
				},
				json=payload,
			)
			handshake_start = time.monotonic()
			response = self._run_with_wall_clock_timeout(
				request_timeout_seconds,
				lambda: client.send(request, stream=True),
			)
			handshake_seconds = round(time.monotonic() - handshake_start, 6)
			response.raise_for_status()
		except Exception:
			client.close()
			raise
		return _RawOpenRouterStreamingResponse(
			client=client,
			response=response,
			handshake_seconds=handshake_seconds,
		)

	def _should_stream_method_synthesis_response(self) -> bool:
		base_url = str(self.base_url or "").strip().lower()
		return "openrouter.ai" in base_url

	def _method_synthesis_request_plugins(
		self,
		stream_response: bool,
		*,
		request_profile: Optional[Dict[str, Any]] = None,
	) -> Optional[List[Dict[str, Any]]]:
		base_url = str(self.base_url or "").strip().lower()
		if stream_response or "openrouter.ai" not in base_url:
			return None
		if not bool((request_profile or {}).get("response_healing_plugin")):
			return None
		return [{"id": "response-healing"}]

	def _openrouter_provider_routing_body(
		self,
		*,
		request_profile: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any] | None:
		base_url = str(self.base_url or "").strip().lower()
		if "openrouter.ai" not in base_url:
			return None
		model_name = str(self.model or "").strip()
		if "/" not in model_name:
			return None
		provider_name = model_name.split("/", 1)[0].strip().lower()
		if not provider_name:
			return None
		extra_body: Dict[str, Any] = {
			"provider": {
				"only": [provider_name],
				"allow_fallbacks": False,
			},
			"session_id": METHOD_LIBRARY_GENERATION_SESSION_ID,
		}
		reasoning_max_tokens = int(
			(request_profile or {}).get("reasoning_max_tokens") or 0,
		)
		extra_body["reasoning"] = {"exclude": False}
		if reasoning_max_tokens > 0:
			extra_body["reasoning"]["max_tokens"] = reasoning_max_tokens
		return extra_body

	def _method_synthesis_request_profile(
		self,
		*,
		prompt: Optional[Dict[str, str]] = None,
	) -> Dict[str, Any]:
		model_name = str(self.model or "").strip().lower()
		if model_name.startswith("moonshotai/"):
			context_window_tokens = self._method_synthesis_total_context_tokens()
			prompt_token_estimate = self._estimate_method_synthesis_prompt_token_budget(prompt)
			return {
				"name": "kimi_stream_single_pass",
				"stream_response": True,
				# Keep reasoning chunks visible as heartbeat events, but cap requested
				# reasoning so final JSON has enough output budget.
				"reasoning_max_tokens": KIMI_REASONING_MAX_TOKENS,
				"first_chunk_timeout_seconds": KIMI_SINGLE_PASS_FIRST_CHUNK_TIMEOUT_SECONDS,
				"response_healing_plugin": False,
				"context_window_tokens": context_window_tokens,
				"prompt_token_estimate": prompt_token_estimate,
				"completion_max_tokens": KIMI_COMPLETION_MAX_TOKENS,
				"reasoning_excluded": False,
				"session_id": METHOD_LIBRARY_GENERATION_SESSION_ID,
				"max_tokens_policy": "fixed_65536",
			}
		return {
			"name": "default_profile",
			"stream_response": self._should_stream_method_synthesis_response(),
			"reasoning_max_tokens": None,
			"first_chunk_timeout_seconds": 0.0,
			"response_healing_plugin": False,
		}

	def _method_synthesis_total_context_tokens(self) -> int:
		base_url = str(self.base_url or "").strip().lower()
		if "openrouter.ai" in base_url:
			return KIMI_OPENROUTER_CONTEXT_WINDOW_TOKENS
		return KIMI_DIRECT_CONTEXT_WINDOW_TOKENS

	@staticmethod
	def _estimate_method_synthesis_prompt_token_budget(
		prompt: Optional[Dict[str, str]],
	) -> int:
		if not isinstance(prompt, dict):
			return 0
		total_characters = sum(len(str(prompt.get(key) or "")) for key in ("system", "user"))
		if total_characters <= 0:
			return 0
		return max(
			1,
			math.ceil(total_characters / KIMI_PROMPT_ESTIMATE_CHARS_PER_TOKEN),
		)

	def _consume_streaming_llm_response(
		self,
		response: object,
		*,
		transport_metadata: Optional[Dict[str, Any]] = None,
		total_timeout_seconds: float = 0.0,
	) -> Tuple[str, Optional[str], Dict[str, Any]]:
		metadata = transport_metadata if transport_metadata is not None else {}
		metadata["llm_response_mode"] = "streaming"
		request_id = self._extract_transport_request_id(response)
		if request_id:
			metadata["llm_request_id"] = request_id
		handshake_seconds = getattr(response, "handshake_seconds", None)
		if handshake_seconds is not None:
			metadata["llm_stream_handshake_seconds"] = handshake_seconds
			self._emit_method_synthesis_progress(
				f"stream_handshake_seconds={handshake_seconds}",
			)
		parts: list[str] = []
		reasoning_chunks_ignored = 0
		finish_reason: Optional[str] = None
		close_stream = getattr(response, "close", None)
		deadline_expired = threading.Event()
		deadline_timer: Optional[threading.Timer] = None

		def _timeout_error() -> TimeoutError:
			error = TimeoutError(
				"Method-synthesis LLM call exceeded the configured timeout before "
				"returning a usable response.",
			)
			try:
				setattr(error, "transport_metadata", dict(metadata))
			except Exception:
				pass
			return error

		def _close_stream_after_deadline() -> None:
			deadline_expired.set()
			if callable(close_stream):
				try:
					close_stream()
				except Exception:
					pass

		stream_start = time.monotonic()
		first_stream_chunk_recorded = False
		first_content_chunk_recorded = False
		first_chunk_timeout_seconds = float(
			metadata.get("llm_first_chunk_timeout_seconds") or 0.0,
		)
		if total_timeout_seconds > 0.0:
			deadline_timer = threading.Timer(
				total_timeout_seconds,
				_close_stream_after_deadline,
			)
			deadline_timer.daemon = True
			deadline_timer.start()
		response_iterator = iter(response)
		try:
			while True:
				elapsed_seconds = time.monotonic() - stream_start
				if deadline_expired.is_set():
					raise _timeout_error()
				remaining_total_timeout_seconds = (
					total_timeout_seconds - elapsed_seconds
					if total_timeout_seconds > 0.0
					else 0.0
				)
				next_chunk_timeout_seconds: Optional[float] = None
				if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0:
					next_chunk_timeout_seconds = first_chunk_timeout_seconds - elapsed_seconds
					if total_timeout_seconds > 0.0:
						next_chunk_timeout_seconds = min(
							next_chunk_timeout_seconds,
							remaining_total_timeout_seconds,
						)
				elif total_timeout_seconds > 0.0:
					next_chunk_timeout_seconds = remaining_total_timeout_seconds
				if next_chunk_timeout_seconds is not None and next_chunk_timeout_seconds <= 0.0:
					timeout_error = TimeoutError(
						"Method-synthesis LLM call exceeded the configured first-chunk "
						"deadline before any streaming chunk arrived."
						if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0
						else "Method-synthesis LLM call exceeded the configured timeout "
						"before returning a usable response.",
					)
					try:
						setattr(timeout_error, "transport_metadata", dict(metadata))
					except Exception:
						pass
					if callable(close_stream):
						close_stream()
					raise timeout_error
				try:
					chunk = self._run_with_wall_clock_timeout(
						next_chunk_timeout_seconds,
						lambda: next(response_iterator),
					)
				except StopIteration:
					if deadline_expired.is_set():
						raise _timeout_error()
					break
				except TimeoutError as exc:
					timeout_error = TimeoutError(
						"Method-synthesis LLM call exceeded the configured first-chunk "
						"deadline before any streaming chunk arrived."
						if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0
						else "Method-synthesis LLM call exceeded the configured timeout "
						"before returning a usable response.",
					)
					try:
						setattr(timeout_error, "transport_metadata", dict(metadata))
					except Exception:
						pass
					if callable(close_stream):
						close_stream()
					raise timeout_error from exc
				except Exception as exc:
					if deadline_expired.is_set():
						raise _timeout_error() from exc
					raise
				request_id = self._extract_transport_request_id(chunk)
				if request_id:
					metadata["llm_request_id"] = request_id
				if not first_stream_chunk_recorded:
					first_stream_chunk_seconds = round(time.monotonic() - stream_start, 6)
					metadata["llm_first_stream_chunk_seconds"] = first_stream_chunk_seconds
					metadata["llm_first_chunk_seconds"] = first_stream_chunk_seconds
					first_stream_chunk_recorded = True
					self._emit_method_synthesis_progress(
						f"first_stream_chunk_seconds={first_stream_chunk_seconds}",
					)
				choices = getattr(chunk, "choices", None) or ()
				if not choices:
					continue
				choice = choices[0]
				finish_reason = getattr(choice, "finish_reason", None) or finish_reason
				delta = getattr(choice, "delta", None)
				for candidate in (
					getattr(delta, "content", None) if delta is not None else None,
					getattr(delta, "parsed", None) if delta is not None else None,
					getattr(choice, "message", None),
				):
					extracted = self._normalise_response_content(candidate)
					if extracted is not None:
						if not first_content_chunk_recorded:
							metadata["llm_first_content_chunk_seconds"] = round(
								time.monotonic() - stream_start,
								6,
							)
							self._emit_method_synthesis_progress(
								"first_content_chunk_seconds="
								f"{metadata['llm_first_content_chunk_seconds']}",
							)
							first_content_chunk_recorded = True
						parts.append(extracted)
				for reasoning_candidate in (
					getattr(delta, "reasoning", None) if delta is not None else None,
					getattr(delta, "reasoning_content", None) if delta is not None else None,
					getattr(delta, "reasoning_details", None) if delta is not None else None,
					getattr(choice, "reasoning", None),
				):
					if reasoning_candidate is None:
						continue
					reasoning_chunks_ignored += 1
					metadata["llm_reasoning_chunks_ignored"] = reasoning_chunks_ignored
					if "llm_first_reasoning_chunk_seconds" not in metadata:
						metadata["llm_first_reasoning_chunk_seconds"] = round(
							time.monotonic() - stream_start,
							6,
						)
				current_text = "".join(parts).strip()
				complete_payload = self._extract_complete_json_payload_text(current_text)
				if complete_payload is not None:
					metadata["llm_complete_json_seconds"] = round(
						time.monotonic() - stream_start,
						6,
					)
					self._emit_method_synthesis_progress(
						f"complete_json_seconds={metadata['llm_complete_json_seconds']}",
					)
					if callable(close_stream):
						close_stream()
					return complete_payload, finish_reason or "stop", dict(metadata)
				if (
					deadline_expired.is_set()
					or (
						total_timeout_seconds > 0.0
						and (time.monotonic() - stream_start) >= total_timeout_seconds
					)
				):
					if callable(close_stream):
						close_stream()
					raise _timeout_error()
		finally:
			if deadline_timer is not None:
				deadline_timer.cancel()

		text = "".join(parts).strip()
		complete_payload = self._extract_complete_json_payload_text(text)
		if complete_payload is not None:
			metadata["llm_complete_json_seconds"] = round(
				time.monotonic() - stream_start,
				6,
			)
			self._emit_method_synthesis_progress(
				f"complete_json_seconds={metadata['llm_complete_json_seconds']}",
			)
			if callable(close_stream):
				close_stream()
			return complete_payload, finish_reason or "stop", dict(metadata)
		if text and any(token in text for token in ("{", "[")):
			if callable(close_stream):
				close_stream()
			return text, finish_reason, dict(metadata)
		if text:
			error = LLMStreamingResponseError(
				"LLM response did not contain usable textual JSON content. "
				f"finish_reason={finish_reason!r}",
				partial_text=text,
				finish_reason=finish_reason,
			)
			try:
				setattr(error, "transport_metadata", dict(metadata))
			except Exception:
				pass
			raise error
		error = LLMStreamingResponseError(
			"LLM response did not contain usable textual JSON content. "
			f"finish_reason={finish_reason!r}",
			finish_reason=finish_reason,
		)
		try:
			setattr(error, "transport_metadata", dict(metadata))
		except Exception:
			pass
		raise error

	def _extract_response_text(self, response: object) -> str:
		choices = getattr(response, "choices", None) or ()
		if not choices:
			response_dump = response.model_dump() if hasattr(response, "model_dump") else None
			if isinstance(response_dump, dict):
				extracted = self._extract_response_text_from_response_dump(response_dump)
				if extracted is not None:
					return extracted
			raise RuntimeError("LLM response did not include any choices.")

		message = getattr(choices[0], "message", None)
		if message is None:
			raise RuntimeError("LLM response choice did not include a message payload.")

		for candidate in (
			getattr(message, "content", None),
			getattr(message, "parsed", None),
		):
			extracted = self._normalise_response_content(candidate)
			if extracted is not None:
				return extracted

		dumped_message = message.model_dump() if hasattr(message, "model_dump") else None
		if isinstance(dumped_message, dict):
			for key in ("content", "parsed", "output_text", "text"):
				extracted = self._normalise_response_content(dumped_message.get(key))
				if extracted is not None:
					return extracted
			refusal = dumped_message.get("refusal")
			refusal_text = self._normalise_response_content(refusal)
			if refusal_text:
				raise RuntimeError(f"LLM refused method-synthesis response: {refusal_text}")

		response_dump = response.model_dump() if hasattr(response, "model_dump") else None
		if isinstance(response_dump, dict):
			extracted = self._extract_response_text_from_response_dump(response_dump)
			if extracted is not None:
				return extracted

		finish_reason = getattr(choices[0], "finish_reason", None)
		raise RuntimeError(
			"LLM response did not contain usable textual JSON content. "
			f"finish_reason={finish_reason!r}",
		)

	@staticmethod
	def _normalise_response_content(content: object) -> str | None:
		if content is None:
			return None
		if isinstance(content, str):
			text = content.strip()
			return text or None
		if isinstance(content, dict):
			for key in ("text", "value", "content"):
				extracted = MethodSynthesisLLMTransportMixin._normalise_response_content(content.get(key))
				if extracted is not None:
					return extracted
			try:
				return json.dumps(content, ensure_ascii=False)
			except TypeError:
				return str(content).strip() or None
		if isinstance(content, (list, tuple)):
			parts: list[str] = []
			for item in content:
				extracted = MethodSynthesisLLMTransportMixin._normalise_response_content(item)
				if extracted is not None:
					parts.append(extracted)
			if not parts:
				return None
			return "\n".join(parts).strip() or None
		text_attr = getattr(content, "text", None)
		extracted = MethodSynthesisLLMTransportMixin._normalise_response_content(text_attr)
		if extracted is not None:
			return extracted
		value_attr = getattr(content, "value", None)
		extracted = MethodSynthesisLLMTransportMixin._normalise_response_content(value_attr)
		if extracted is not None:
			return extracted
		stringified = str(content).strip()
		return stringified or None

	@classmethod
	def _extract_response_text_from_response_dump(cls, response_dump: Dict[str, Any]) -> str | None:
		choices = response_dump.get("choices")
		if isinstance(choices, list) and choices:
			first_choice = choices[0]
			if isinstance(first_choice, dict):
				message = first_choice.get("message")
				if isinstance(message, dict):
					for key in ("content", "parsed", "output_text", "text"):
						extracted = cls._normalise_response_content(message.get(key))
						if extracted is not None:
							return extracted
					if any(key in message for key in ("target_task_bindings", "tasks", "compound_tasks", "methods")):
						extracted = cls._normalise_response_content(message)
						if extracted is not None:
							return extracted
				for key in ("content", "parsed", "output_text", "text"):
					extracted = cls._normalise_response_content(first_choice.get(key))
					if extracted is not None:
						return extracted
		for key in ("output_text", "text", "content", "parsed"):
			extracted = cls._normalise_response_content(response_dump.get(key))
			if extracted is not None:
				return extracted
		return None

	@staticmethod
	def _extract_transport_request_id(payload: object) -> str | None:
		for attr_name in ("id", "response_id", "request_id", "_request_id"):
			value = getattr(payload, attr_name, None)
			if isinstance(value, str) and value.strip():
				return value.strip()
		response_payload = getattr(payload, "response", None)
		if response_payload is not None:
			for headers_attr in ("headers", "_headers"):
				headers = getattr(response_payload, headers_attr, None)
				if headers is None:
					continue
				for key in ("x-request-id", "request-id", "openai-request-id"):
					try:
						value = headers.get(key)
					except Exception:
						value = None
					if isinstance(value, str) and value.strip():
						return value.strip()
		return None

	@classmethod
	def _extract_complete_json_payload_text(cls, text: str) -> str | None:
		stripped = str(text or "").strip()
		if not stripped:
			return None

		def _decode_from(start_index: int) -> str | None:
			candidate = stripped[start_index:]
			if cls._appears_truncated_json(candidate):
				return None
			try:
				parsed, end_index = json.JSONDecoder().raw_decode(candidate)
			except json.JSONDecodeError:
				return None
			if not isinstance(parsed, (dict, list)):
				return None
			payload = candidate[:end_index].strip()
			return payload or None

		first_nonspace = stripped[0]
		if first_nonspace in "{[":
			payload = _decode_from(0)
			if payload is not None:
				return payload

		object_index = stripped.find("{")
		if object_index != -1:
			payload = _decode_from(object_index)
			return payload

		array_index = stripped.find("[")
		if array_index != -1:
			payload = _decode_from(array_index)
			return payload
		return None

	def _parse_llm_library(
		self,
		response_text: str,
		*,
		ast_compiler_defaults: Optional[Dict[str, Any]] = None,
	) -> HTNMethodLibrary:
		clean_text = self._strip_code_fences(response_text)
		salvaged_tail_payload = self._salvage_missing_object_closer_at_tail(clean_text)
		if self._appears_truncated_json(clean_text):
			if salvaged_tail_payload is not None:
				payload = salvaged_tail_payload
			else:
				raise ValueError(
					"LLM response appears truncated before the JSON object closed. "
					"The HTN library was cut off mid-response.",
				)
		else:
			try:
				payload = json.loads(clean_text)
			except json.JSONDecodeError as original_error:
				common_quoting_repair_payload = self._salvage_common_json_quoting_errors(
					clean_text,
				)
				if common_quoting_repair_payload is not None:
					payload = common_quoting_repair_payload
				else:
					salvaged_payload = self._salvage_ast_payload(clean_text)
					if salvaged_payload is not None:
						payload = salvaged_payload
					else:
						salvaged_domain_payload = self._salvage_domain_task_payload(clean_text)
						if salvaged_domain_payload is not None:
							payload = salvaged_domain_payload
						else:
							missing_object_closer_payload = self._salvage_missing_object_closer(
								clean_text,
								original_error,
							)
							if missing_object_closer_payload is not None:
								payload = missing_object_closer_payload
							else:
								raw_decoded = self._decode_leading_json_object(clean_text)
								if raw_decoded is not None:
									payload = raw_decoded
								else:
									candidate = self._extract_json_object_candidate(clean_text)
									if candidate is None:
										raise ValueError(
											f"HTN synthesis response could not be parsed as JSON: {original_error}"
										) from original_error
									try:
										payload = json.loads(candidate)
									except json.JSONDecodeError as candidate_error:
										raise ValueError(
											"HTN synthesis response could not be parsed as JSON: "
											f"{candidate_error}"
										) from original_error
		if isinstance(payload, list):
			if (
				len(payload) == 1
				and isinstance(payload[0], dict)
				and isinstance(payload[0].get("tasks"), list)
			):
				payload = payload[0]
			elif all(isinstance(item, dict) for item in payload):
				payload = {"tasks": payload}
			else:
				raise ValueError("HTN synthesis response must be a JSON object")
		if not isinstance(payload, dict):
			raise ValueError("HTN synthesis response must be a JSON object")
		if ast_compiler_defaults:
			payload = self._apply_ast_compiler_defaults(
				payload,
				ast_compiler_defaults=ast_compiler_defaults,
			)
		return HTNMethodLibrary.from_dict(payload)

	@classmethod
	def _salvage_domain_task_payload(cls, result_text: str) -> dict | None:
		task_object_texts = cls._extract_named_task_object_fragments(result_text)
		if not task_object_texts:
			return None
		tasks: List[Dict[str, Any]] = []
		for task_text in task_object_texts:
			try:
				task_payload = json.loads(task_text)
			except json.JSONDecodeError:
				continue
			if not isinstance(task_payload, dict):
				continue
			if not str(task_payload.get("name", "")).strip():
				continue
			tasks.append(task_payload)
		if not tasks:
			return None
		return {"tasks": tasks}

	@staticmethod
	def _salvage_common_json_quoting_errors(text: str) -> Optional[Dict[str, Any] | List[Any]]:
		repaired_text = str(text or "")
		repaired_text = re.sub(r'([\]}])"\s*,\s*"', r'\1,"', repaired_text)
		repaired_text = re.sub(
			r'(?<=[}\]])\s*,\s*"name"\s*:',
			r',{"name":',
			repaired_text,
		)
		repaired_text = re.sub(
			r'("[A-Za-z0-9_-]+"\s*:\s*"[^"\r\n]*?\))(?=\s*[,}\]])',
			r'\1"',
			repaired_text,
		)
		if repaired_text == str(text or ""):
			return None
		try:
			return json.loads(repaired_text)
		except json.JSONDecodeError:
			return None
