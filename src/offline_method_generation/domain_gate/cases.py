"""
Case models for domain-gate validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DomainGateCase:
	"""One synthetic validation case for a compound task."""

	task_name: str
	task_args: Tuple[str, ...]
	object_types: Dict[str, str]
	object_pool: Tuple[str, ...]
