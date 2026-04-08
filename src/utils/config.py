"""
Configuration Management

Handles loading environment variables and API keys from .env file.
"""

import os
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_STAGE1_MODEL = "deepseek/deepseek-chat-v3-0324"
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

        Stage 3 needs a model that can return one structured JSON library within
        the completion budget. Some reasoning-first endpoints reserve the full
        completion budget for hidden reasoning and never emit final JSON. Allow
        Stage 3 to use a separate model from Stage 1 while keeping the default
        backward-compatible.
        """
        return os.getenv('OPENAI_STAGE3_MODEL', self.openai_model)

    @property
    def openai_timeout(self) -> int:
        """Get API timeout in seconds (default: 30)"""
        return int(os.getenv('OPENAI_TIMEOUT', '30'))

    @property
    def openai_stage3_timeout(self) -> int:
        """
        Get the Stage 3 request timeout in seconds.

        Stage 3 synthesizes an entire method library in one response, so it needs a
        looser default wall-clock budget than Stage 1's semantic-parsing request.
        """
        return int(os.getenv('OPENAI_STAGE3_TIMEOUT', '180'))

    @property
    def openai_stage3_max_tokens(self) -> int:
        """
        Get the Stage 3 response token budget.

        Stage 3 now emits a narrow AST-style JSON library rather than free-form text.
        Keep the default well below the old 20000-token budget to reduce latency
        and to stay within common structured-output limits such as DeepSeek's 8K
        maximum for `deepseek-chat`.
        """
        return max(int(os.getenv('OPENAI_STAGE3_MAX_TOKENS', '8000')), 1)

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
