"""
Configuration Management

Handles loading environment variables and API keys from .env file.
"""

import os
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_STAGE1_MODEL = "deepseek/deepseek-chat-v3-0324"
DEFAULT_STAGE3_MODEL = "minimax/minimax-m2"
DEFAULT_SHARED_MODEL = "deepseek-chat"


class Config:
    """Configuration manager for the LLM-BDI Pipeline"""

    def __init__(self):
        self._load_env()

    def _load_env(self):
        """Load environment variables from .env file"""
        # Path calculation: utils/ -> src/ -> project_root/
        env_path = Path(__file__).parent.parent.parent / ".env"

        if env_path.exists():
            with open(env_path, 'r') as f:
                self._merge_env_lines(f)

    def _merge_env_lines(self, lines: Iterable[str]) -> None:
        """
        Merge dotenv-style lines without overriding explicit shell environment.

        This keeps `.env` as a repository-local default source while preserving
        per-run overrides such as benchmark model selection exported in the
        invoking shell.
        """
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY')

    @property
    def openai_model(self) -> str:
        """Get OpenAI model (default: deepseek-chat)"""
        return os.getenv('OPENAI_MODEL', DEFAULT_SHARED_MODEL)

    @property
    def openai_stage1_model(self) -> str:
        """
        Get the Stage 1 model identifier.

        Stage 1 must not inherit the shared model selection used by other stages.
        Keep it pinned to the DeepSeek chat model family by default so benchmark
        runs remain stable even when Stage 3 experiments temporarily switch the
        shared model to another provider.
        """
        return os.getenv('OPENAI_STAGE1_MODEL', DEFAULT_STAGE1_MODEL)

    @property
    def openai_stage3_model(self) -> str:
        """
        Get the Stage 3 model identifier.

        Stage 3 needs a Minimax model that can return one structured JSON
        library within a bounded wall-clock budget. The smaller M2 variant is a
        better fit for low-latency one-shot structured generation than the
        slower reasoning-heavier variants.
        """
        return os.getenv('OPENAI_STAGE3_MODEL', DEFAULT_STAGE3_MODEL)

    @property
    def openai_timeout(self) -> int:
        """Get API timeout in seconds (default: 30)"""
        return int(os.getenv('OPENAI_TIMEOUT', '30'))

    @property
    def openai_stage3_timeout(self) -> int:
        """
        Get the Stage 3 request timeout in seconds.

        Stage 3 synthesizes an entire transition-native method library in one
        Minimax response. The compact AST prompt reduces output size, but the
        model can still need materially longer wall-clock time than Stage 1 on
        larger ordered task networks while reasoning remains enabled.
        """
        return int(os.getenv('OPENAI_STAGE3_TIMEOUT', '600'))

    @property
    def openai_stage3_max_tokens(self) -> int:
        """
        Get the Stage 3 response token budget.

        Stage 3 emits one complete method library in a single Minimax response.
        The transition-native redesign keeps the JSON compact, but the one-shot
        library can still be materially larger than Stage 1's symbolic output,
        especially when reasoning remains enabled. Keep a larger one-shot budget
        by default so the model can finish a full library on longer benchmark
        queries without requiring retries or repairs.
        """
        return max(int(os.getenv('OPENAI_STAGE3_MAX_TOKENS', '48000')), 1)

    @property
    def openai_stage1_max_tokens(self) -> int:
        """
        Get the Stage 1 response token budget.

        Stage 1 can receive long benchmark queries that enumerate many task
        invocations. Leaving output length to provider defaults can truncate the
        returned JSON object even when the prompt is otherwise valid.
        """
        return max(int(os.getenv('OPENAI_STAGE1_MAX_TOKENS', '12000')), 1)

    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL (for custom endpoints like DeepSeek)"""
        return os.getenv('OPENAI_BASE_URL')

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            return False
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config
