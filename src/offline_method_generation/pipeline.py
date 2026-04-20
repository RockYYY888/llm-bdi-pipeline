"""
Offline method-generation pipeline entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pipeline.execution_logger import ExecutionLogger

from .context import OfflineSynthesisContext
from .orchestrator import OfflineDomainSynthesisOrchestrator


class OfflineMethodGenerationPipeline:
	"""Public offline entrypoint for generated domain-library construction."""

	def __init__(self, *, domain_file: str) -> None:
		self._context = OfflineSynthesisContext(domain_file=domain_file)
		self._orchestrator = OfflineDomainSynthesisOrchestrator(self._context)

	@property
	def context(self) -> OfflineSynthesisContext:
		"""Expose the offline-only synthesis context."""
		return self._context

	@property
	def logger(self) -> ExecutionLogger:
		return self._context.logger

	@logger.setter
	def logger(self, value: ExecutionLogger) -> None:
		self._context.logger = value

	def build_domain_library(
		self,
		*,
		output_root: Optional[str] = None,
	) -> Dict[str, Any]:
		return self._orchestrator.build_domain_library(output_root=output_root)
