"""
Evaluation domain-source selection for plan-library execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


EVALUATION_DOMAIN_SOURCE_BENCHMARK = "benchmark"
EVALUATION_DOMAIN_SOURCE_GENERATED = "generated"
SUPPORTED_EVALUATION_DOMAIN_SOURCES = (
	EVALUATION_DOMAIN_SOURCE_BENCHMARK,
	EVALUATION_DOMAIN_SOURCE_GENERATED,
)


def normalize_evaluation_domain_source(source: Optional[str]) -> str:
	"""Validate and normalize the configured evaluation domain source."""
	normalized = str(source or EVALUATION_DOMAIN_SOURCE_BENCHMARK).strip().lower()
	if normalized not in SUPPORTED_EVALUATION_DOMAIN_SOURCES:
		raise ValueError(
			"Unsupported evaluation domain source "
			f"'{source}'. Expected one of {SUPPORTED_EVALUATION_DOMAIN_SOURCES}.",
		)
	return normalized


@dataclass(frozen=True)
class EvaluationDomainContext:
	"""Resolved evaluation runtime domain configuration."""

	source: str
	domain_file: str
	domain: Any
	type_parent_map: Dict[str, Optional[str]]
	predicate_type_map: Dict[str, Tuple[str, ...]]
	action_type_map: Dict[str, Tuple[str, ...]]
	task_type_map: Dict[str, Tuple[str, ...]]
