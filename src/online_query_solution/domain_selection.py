"""
Online domain-source selection for query execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


ONLINE_DOMAIN_SOURCE_BENCHMARK = "benchmark"
ONLINE_DOMAIN_SOURCE_GENERATED = "generated"
SUPPORTED_ONLINE_DOMAIN_SOURCES = (
	ONLINE_DOMAIN_SOURCE_BENCHMARK,
	ONLINE_DOMAIN_SOURCE_GENERATED,
)


def normalize_online_domain_source(source: Optional[str]) -> str:
	"""Validate and normalize the configured online domain source."""
	normalized = str(source or ONLINE_DOMAIN_SOURCE_BENCHMARK).strip().lower()
	if normalized not in SUPPORTED_ONLINE_DOMAIN_SOURCES:
		raise ValueError(
			"Unsupported online domain source "
			f"'{source}'. Expected one of {SUPPORTED_ONLINE_DOMAIN_SOURCES}.",
		)
	return normalized


@dataclass(frozen=True)
class OnlineDomainContext:
	"""Resolved online runtime domain configuration."""

	source: str
	domain_file: str
	domain: Any
	type_parent_map: Dict[str, Optional[str]]
	predicate_type_map: Dict[str, Tuple[str, ...]]
	action_type_map: Dict[str, Tuple[str, ...]]
	task_type_map: Dict[str, Tuple[str, ...]]

