"""Language-model transport and response extraction for method synthesis."""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

from language_model import (
	OPENAI_COMPATIBLE_JSON_PROFILE_NAME,
	create_openai_compatible_json_completion,
)
from method_library.synthesis.schema import HTNMethodLibrary
from .errors import LLMStreamingResponseError


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

	def _apply_method_synthesis_provider_token_ceiling(
		self,
		requested_max_tokens: int | None,
	) -> int | None:
		if requested_max_tokens is None:
			return None
		return max(int(requested_max_tokens), 1)

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
			"llm_first_chunk_timeout_seconds": request_profile.get("first_chunk_timeout_seconds"),
		}
		for metadata_key, profile_key in (
			("llm_completion_max_tokens", "completion_max_tokens"),
			("llm_max_tokens_policy", "max_tokens_policy"),
			("llm_thinking_type", "thinking_type"),
			("llm_reasoning_effort", "reasoning_effort"),
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
		stream_response = bool(profile.get("stream_response", False))
		messages = [
			{"role": "system", "content": prompt["system"]},
			{"role": "user", "content": prompt["user"]},
		]
		return self._run_with_wall_clock_timeout(
			request_timeout_seconds,
			lambda: create_openai_compatible_json_completion(
				self.client,
				model=self.model,
				messages=messages,
				timeout=request_timeout_seconds
				if request_timeout_seconds is not None
				else self.timeout,
				max_tokens=max_tokens,
				stream=stream_response,
				reasoning_effort=profile.get("reasoning_effort", "max"),
				thinking_type=profile.get("thinking_type", "enabled"),
			),
		)

	def _method_synthesis_request_profile(
		self,
		*,
		prompt: Optional[Dict[str, str]] = None,
	) -> Dict[str, Any]:
		_ = prompt
		return {
			"name": OPENAI_COMPATIBLE_JSON_PROFILE_NAME,
			"stream_response": False,
			"first_chunk_timeout_seconds": 0.0,
			"completion_max_tokens": max(int(getattr(self, "max_tokens", 0) or 0), 1),
			"max_tokens_policy": "configured_method_synthesis_max_tokens",
			"thinking_type": "enabled",
			"reasoning_effort": "max",
		}

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
