"""
Online query-solution pipeline entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict

from pipeline.domain_complete_pipeline import DomainCompletePipeline


class OnlineQuerySolutionPipeline:
	"""Public online entrypoint for Jason-based natural-language query solving."""

	def __init__(
		self,
		*,
		domain_file: str,
		problem_file: str | None = None,
		online_domain_source: str | None = None,
	) -> None:
		self._pipeline = DomainCompletePipeline(
			domain_file=domain_file,
			problem_file=problem_file,
			online_domain_source=online_domain_source,
		)

	@property
	def pipeline(self) -> DomainCompletePipeline:
		"""Expose the underlying compatibility pipeline when low-level access is needed."""
		return self._pipeline

	def run_query(self, nl_instruction: str) -> Dict[str, Any]:
		return self._pipeline.run_query(nl_instruction)

	def execute_query_with_library(
		self,
		nl_query: str,
		*,
		library_artifact: Any,
	) -> Dict[str, Any]:
		return self._pipeline.execute_query_with_library(
			nl_query,
			library_artifact=library_artifact,
		)
