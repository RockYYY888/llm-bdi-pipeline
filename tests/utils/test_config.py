"""
Configuration tests for shared and Stage 3-specific API settings.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import Config


def test_stage3_config_falls_back_to_shared_openai_settings(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.setenv("OPENAI_MODEL", "shared-model")
    monkeypatch.setenv("OPENAI_TIMEOUT", "42")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://shared.example.com")
    monkeypatch.delenv("STAGE3_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STAGE3_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("STAGE3_OPENAI_TIMEOUT", raising=False)
    monkeypatch.delenv("STAGE3_OPENAI_BASE_URL", raising=False)

    config = Config()

    assert config.stage3_openai_api_key == "sk-shared"
    assert config.stage3_openai_model == "shared-model"
    assert config.stage3_openai_timeout == 42
    assert config.stage3_openai_base_url == "https://shared.example.com"


def test_stage3_config_can_override_shared_openai_settings(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.setenv("OPENAI_MODEL", "shared-model")
    monkeypatch.setenv("OPENAI_TIMEOUT", "42")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://shared.example.com")
    monkeypatch.setenv("STAGE3_OPENAI_API_KEY", "sk-stage3")
    monkeypatch.setenv("STAGE3_OPENAI_MODEL", "stage3-model")
    monkeypatch.setenv("STAGE3_OPENAI_TIMEOUT", "99")
    monkeypatch.setenv("STAGE3_OPENAI_BASE_URL", "https://stage3.example.com")

    config = Config()

    assert config.stage3_openai_api_key == "sk-stage3"
    assert config.stage3_openai_model == "stage3-model"
    assert config.stage3_openai_timeout == 99
    assert config.stage3_openai_base_url == "https://stage3.example.com"
