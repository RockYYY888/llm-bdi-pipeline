"""
Offline method-generation pipeline entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pipeline.domain_complete_pipeline import DomainCompletePipeline


class OfflineMethodGenerationPipeline:
	"""Public offline entrypoint for generated domain-library construction."""

	def __init__(self, *, domain_file: str) -> None:
		self._pipeline = DomainCompletePipeline(domain_file=domain_file)

	@property
	def pipeline(self) -> DomainCompletePipeline:
		"""Expose the underlying compatibility pipeline when low-level access is needed."""
		return self._pipeline

	def build_domain_library(
		self,
		*,
		output_root: Optional[str] = None,
	) -> Dict[str, Any]:
		return self._pipeline.build_domain_library(output_root=output_root)
