"""Method-synthesis error types."""

from __future__ import annotations

from typing import Any, Dict, Optional


class HTNSynthesisError(RuntimeError):
	"""Raised when method synthesis cannot produce a valid HTN method library."""

	def __init__(
		self,
		message: str,
		*,
		model: Optional[str],
		llm_prompt: Optional[Dict[str, str]],
		llm_response: Optional[str],
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		super().__init__(message)
		self.model = model
		self.llm_prompt = llm_prompt
		self.llm_response = llm_response
		self.metadata = dict(metadata or {})


class LLMStreamingResponseError(RuntimeError):
	"""Raised when a streaming synthesis response is unusable but diagnostically valuable."""

	def __init__(
		self,
		message: str,
		*,
		partial_text: Optional[str] = None,
		finish_reason: Optional[str] = None,
	) -> None:
		super().__init__(message)
		self.partial_text = partial_text
		self.finish_reason = finish_reason
