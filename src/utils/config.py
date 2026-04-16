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

        Stage 1 is part of the benchmark contract, not a per-shell tuning knob.
        Keep it pinned to the DeepSeek chat model unconditionally so live query
        evaluation cannot silently drift to a different provider or reasoning
        variant via shell or dotenv overrides.
        """
        return DEFAULT_STAGE1_MODEL

    @property
    def openai_stage3_model(self) -> str:
        """
        Get the Stage 3 model identifier.

        Stage 3 is benchmark-pinned. The domain-complete one-shot synthesis
        path is evaluated specifically on Minimax M2, so configuration
        overrides must not silently swap in another provider or unrelated
        chat model.
        """
        return DEFAULT_STAGE3_MODEL

    @property
    def openai_timeout(self) -> int:
        """Get API timeout in seconds (default: 30)"""
        return int(os.getenv('OPENAI_TIMEOUT', '30'))

    @property
    def openai_stage3_timeout(self) -> int:
        """
        Get the Stage 3 request timeout in seconds.

        Stage 3 synthesizes one domain-complete method library in a single
        model response. That whole-domain JSON can take materially longer than
        Stage 1 on larger domains.
        """
        return int(os.getenv('OPENAI_STAGE3_TIMEOUT', '600'))

    @property
    def openai_stage3_max_tokens(self) -> int:
        """
        Get the Stage 3 response token budget.

        Stage 3 emits one whole-domain method library in a single response.
        Keep a large one-shot budget by default so the model can finish that
        JSON library without retries or repairs.
        """
        return max(int(os.getenv('OPENAI_STAGE3_MAX_TOKENS', '48000')), 1)

    @property
    def stage6_jason_timeout(self) -> int:
        """
        Get the Stage 6 Jason runtime timeout in seconds.

        Large benchmark queries can require thousands of primitive actions in the
        BDI runtime. Keep this separate from LLM timeouts so runtime validation
        can scale without changing Stage 1 or Stage 3 synthesis behavior.
        """
        return max(int(os.getenv('STAGE6_JASON_TIMEOUT', '600')), 1)

    @property
    def stage5_planning_timeout(self) -> int:
        """
        Get the Stage 5 Hierarchical Task Network planner timeout in seconds.

        The planner is now the primary benchmark-time solver. Large grounded
        request networks must fail closed within a bounded runtime so masked and
        live sweeps can finish and report per-query failure causes instead of
        hanging indefinitely on one hard instance. The default should be large
        enough for full official baselines, not just short smoke runs.
        """
        return max(int(os.getenv('STAGE5_PLANNING_TIMEOUT', '600')), 1)

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
