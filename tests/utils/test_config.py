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
    monkeypatch.setenv("OPENAI_STAGE1_MODEL", "deepseek/deepseek-chat-v3-0324")
    monkeypatch.setenv("OPENAI_STAGE3_MODEL", "deepseek/deepseek-chat-v3-0324")
    monkeypatch.setenv("OPENAI_TIMEOUT", "120")
    monkeypatch.setenv("OPENAI_STAGE3_TIMEOUT", "240")
    monkeypatch.setenv("OPENAI_STAGE3_MAX_TOKENS", "4096")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    config = Config()

    assert config.openai_api_key == "sk-shared"
    assert config.openai_model == "deepseek-chat"
    assert config.openai_stage1_model == "deepseek/deepseek-chat-v3-0324"
    assert config.openai_stage3_model == "deepseek/deepseek-chat-v3-0324"
    assert config.openai_timeout == 120
    assert config.openai_stage3_timeout == 240
    assert config.openai_stage3_max_tokens == 4096
    assert config.openai_base_url == "https://api.deepseek.com"


def test_stage3_max_tokens_defaults_to_structured_output_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("OPENAI_STAGE3_MAX_TOKENS", raising=False)

    config = Config()

    assert config.openai_stage3_max_tokens == 8000


def test_stage1_model_defaults_to_pinned_deepseek_chat(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "shared-model")
    monkeypatch.delenv("OPENAI_STAGE1_MODEL", raising=False)

    config = Config()

    assert config.openai_stage1_model == "deepseek/deepseek-chat-v3-0324"


def test_stage3_model_defaults_to_shared_model(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "shared-model")
    monkeypatch.delenv("OPENAI_STAGE3_MODEL", raising=False)

    config = Config()

    assert config.openai_stage3_model == "shared-model"


def test_dotenv_merge_preserves_explicit_shell_overrides(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    monkeypatch.setenv("OPENAI_STAGE1_MODEL", "shell-stage1-model")
    monkeypatch.delenv("OPENAI_STAGE3_MODEL", raising=False)

    config = Config()
    config._merge_env_lines(
        [
            "OPENAI_MODEL=file-model",
            "OPENAI_STAGE1_MODEL=file-stage1-model",
            "OPENAI_STAGE3_MODEL=file-stage3-model",
        ]
    )

    assert config.openai_model == "shell-model"
    assert config.openai_stage1_model == "shell-stage1-model"
    assert config.openai_stage3_model == "file-stage3-model"
