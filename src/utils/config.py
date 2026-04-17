"""
Configuration management for the domain-complete HTN pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_GOAL_GROUNDING_MODEL = "deepseek/deepseek-chat-v3-0324"
DEFAULT_METHOD_SYNTHESIS_MODEL = "minimax/minimax-m2"
DEFAULT_SHARED_MODEL = "deepseek-chat"


class Config:
	"""Configuration manager for the domain-complete HTN pipeline."""

	def __init__(self) -> None:
		self._load_env()

	def _load_env(self) -> None:
		env_path = Path(__file__).parent.parent.parent / ".env"
		if env_path.exists():
			with open(env_path, "r") as handle:
				self._merge_env_lines(handle)

	def _merge_env_lines(self, lines: Iterable[str]) -> None:
		"""
		Merge dotenv-style lines without overriding explicit shell environment.
		"""
		for raw_line in lines:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			key, value = line.split("=", 1)
			os.environ.setdefault(key.strip(), value.strip())

	@property
	def openai_api_key(self) -> Optional[str]:
		return os.getenv("OPENAI_API_KEY")

	@property
	def openai_model(self) -> str:
		return os.getenv("OPENAI_MODEL", DEFAULT_SHARED_MODEL)

	@property
	def goal_grounding_model(self) -> str:
		"""
		Get the goal-grounding model identifier.

		The query grounding path is benchmark-pinned and should not silently drift
		between providers.
		"""
		return os.getenv("GOAL_GROUNDING_MODEL", DEFAULT_GOAL_GROUNDING_MODEL)

	@property
	def method_synthesis_model(self) -> str:
		"""
		Get the method-synthesis model identifier.

		The domain-complete synthesis path is benchmark-pinned and should remain
		stable across runs.
		"""
		return os.getenv("METHOD_SYNTHESIS_MODEL", DEFAULT_METHOD_SYNTHESIS_MODEL)

	@property
	def openai_timeout(self) -> int:
		return int(os.getenv("OPENAI_TIMEOUT", "30"))

	@property
	def method_synthesis_timeout(self) -> int:
		return max(int(os.getenv("METHOD_SYNTHESIS_TIMEOUT", "600")), 1)

	@property
	def method_synthesis_max_tokens(self) -> int:
		return max(int(os.getenv("METHOD_SYNTHESIS_MAX_TOKENS", "48000")), 1)

	@property
	def planning_timeout(self) -> int:
		return max(int(os.getenv("PLANNING_TIMEOUT", "600")), 1)

	@property
	def goal_grounding_max_tokens(self) -> int:
		return max(int(os.getenv("GOAL_GROUNDING_MAX_TOKENS", "12000")), 1)

	@property
	def openai_base_url(self) -> Optional[str]:
		return os.getenv("OPENAI_BASE_URL")

	def validate(self) -> bool:
		if not self.openai_api_key:
			return False
		if not self.openai_api_key.startswith("sk-"):
			return False
		return True


config = Config()


def get_config() -> Config:
	"""Get the global configuration instance."""
	return config
