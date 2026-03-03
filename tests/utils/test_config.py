"""
Configuration tests for the shared OpenAI settings.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import Config


def test_shared_openai_config_reads_expected_fields(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_TIMEOUT", "120")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    config = Config()

    assert config.openai_api_key == "sk-shared"
    assert config.openai_model == "deepseek-chat"
    assert config.openai_timeout == 120
    assert config.openai_base_url == "https://api.deepseek.com"
