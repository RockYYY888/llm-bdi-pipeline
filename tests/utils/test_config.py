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
    monkeypatch.setenv("OFFLINE_OPENAI_API_KEY", "sk-offline")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("GOAL_GROUNDING_MODEL", "deepseek/deepseek-chat-v3-0324")
    monkeypatch.setenv("METHOD_SYNTHESIS_MODEL", "legacy/provider-model")
    monkeypatch.setenv("OPENAI_TIMEOUT", "120")
    monkeypatch.setenv("METHOD_SYNTHESIS_TIMEOUT", "240")
    monkeypatch.setenv("METHOD_SYNTHESIS_MAX_TOKENS", "4096")
    monkeypatch.setenv("PLANNING_TIMEOUT", "900")
    monkeypatch.setenv("GOAL_GROUNDING_MAX_TOKENS", "2048")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("ONLINE_DOMAIN_SOURCE", "generated")

    config = Config()

    assert config.openai_api_key == "sk-shared"
    assert config.offline_openai_api_key == "sk-offline"
    assert config.openai_model == "deepseek-chat"
    assert config.goal_grounding_model == "deepseek/deepseek-chat-v3-0324"
    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"
    assert config.openai_timeout == 120
    assert config.method_synthesis_timeout == 240
    assert config.method_synthesis_max_tokens == 4096
    assert config.planning_timeout == 900
    assert config.goal_grounding_max_tokens == 2048
    assert config.openai_base_url == "https://api.deepseek.com"
    assert config.online_domain_source == "generated"


def test_method_synthesis_max_tokens_defaults_to_one_shot_library_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_MAX_TOKENS", raising=False)

    config = Config()

    assert config.method_synthesis_max_tokens == 48000


def test_method_synthesis_timeout_defaults_to_longer_one_shot_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_TIMEOUT", raising=False)

    config = Config()

    assert config.method_synthesis_timeout == 1000


def test_planning_timeout_defaults_to_large_runtime_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("PLANNING_TIMEOUT", raising=False)

    config = Config()

    assert config.planning_timeout == 600


def test_goal_grounding_model_inherits_shared_model_when_no_explicit_override(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "moonshotai/kimi-k2.6")
    monkeypatch.delenv("GOAL_GROUNDING_MODEL", raising=False)

    config = Config()

    assert config.goal_grounding_model == "moonshotai/kimi-k2.6"


def test_goal_grounding_model_defaults_to_pinned_kimi_when_no_env_is_set(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("GOAL_GROUNDING_MODEL", raising=False)

    config = Config()

    assert config.goal_grounding_model == "moonshotai/kimi-k2.6"


def test_method_synthesis_model_defaults_to_pinned_kimi_chat(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "shared-model")
    monkeypatch.delenv("METHOD_SYNTHESIS_MODEL", raising=False)

    config = Config()

    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"


def test_offline_api_key_falls_back_to_shared_openai_api_key(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.delenv("OFFLINE_OPENAI_API_KEY", raising=False)

    config = Config()

    assert config.offline_openai_api_key == "sk-shared"


def test_dotenv_merge_preserves_explicit_shell_overrides(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    monkeypatch.setenv("GOAL_GROUNDING_MODEL", "shell-goal-grounding-model")
    monkeypatch.setenv("METHOD_SYNTHESIS_MODEL", "shell-method-synthesis-model")

    config = Config()
    config._merge_env_lines(
        [
            "OPENAI_MODEL=file-model",
            "GOAL_GROUNDING_MODEL=file-goal-grounding-model",
            "METHOD_SYNTHESIS_MODEL=file-method-synthesis-model",
        ]
    )

    assert config.openai_model == "shell-model"
    assert config.goal_grounding_model == "shell-goal-grounding-model"
    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"


def test_method_synthesis_model_is_pinned_even_when_dotenv_sets_legacy_override(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)

    config = Config()
    config._merge_env_lines(
        [
            "METHOD_SYNTHESIS_MODEL=legacy/provider-model",
        ]
    )

    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"


def test_online_domain_source_defaults_to_benchmark(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("ONLINE_DOMAIN_SOURCE", raising=False)

    config = Config()

    assert config.online_domain_source == "benchmark"
