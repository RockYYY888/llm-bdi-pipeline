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
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL (for custom endpoints like DeepSeek)"""
        return os.getenv('OPENAI_BASE_URL')

    @property
    def stage3_openai_api_key(self) -> Optional[str]:
        """Get Stage 3 OpenAI API key, falling back to the shared key."""
        return os.getenv('STAGE3_OPENAI_API_KEY') or self.openai_api_key

    @property
    def stage3_openai_model(self) -> str:
        """Get Stage 3 model, falling back to the shared model."""
        return os.getenv('STAGE3_OPENAI_MODEL', self.openai_model)

    @property
    def stage3_openai_timeout(self) -> int:
        """Get Stage 3 API timeout, falling back to the shared timeout."""
        return int(os.getenv('STAGE3_OPENAI_TIMEOUT', str(self.openai_timeout)))

    @property
    def stage3_openai_base_url(self) -> Optional[str]:
        """Get Stage 3 base URL, falling back to the shared base URL."""
        return os.getenv('STAGE3_OPENAI_BASE_URL') or self.openai_base_url

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
