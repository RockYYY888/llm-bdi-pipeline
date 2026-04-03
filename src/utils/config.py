"""
Configuration Management

Handles loading environment variables and API keys from .env file.
"""

import os
from pathlib import Path
from typing import Optional


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
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY')

    @property
    def openai_model(self) -> str:
        """Get OpenAI model (default: deepseek-chat)"""
        return os.getenv('OPENAI_MODEL', 'deepseek-chat')

    @property
    def openai_timeout(self) -> int:
        """Get API timeout in seconds (default: 30)"""
        return int(os.getenv('OPENAI_TIMEOUT', '30'))

    @property
    def openai_stage3_max_tokens(self) -> int:
        """
        Get the Stage 3 response token budget.

        Stage 3 is a single-shot synthesis step, so repository-level hard clamping
        creates avoidable truncation risk on providers that support larger outputs.
        Leave the budget provider-configurable instead of forcing an 8192-token cap.
        """
        return max(int(os.getenv('OPENAI_STAGE3_MAX_TOKENS', '20000')), 1)

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
