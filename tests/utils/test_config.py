"""
Configuration tests for stage-specific language-model settings.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import Config


def test_stage_specific_generation_config_reads_expected_fields(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.setenv("METHOD_SYNTHESIS_API_KEY", "sk-method")
    monkeypatch.setenv("LTLF_GENERATION_MODEL", "deepseek/deepseek-chat-v3-0324")
    monkeypatch.setenv("METHOD_SYNTHESIS_MODEL", "moonshotai/kimi-k2.6")
    monkeypatch.setenv("LTLF_GENERATION_TIMEOUT", "120")
    monkeypatch.setenv("METHOD_SYNTHESIS_TIMEOUT", "240")
    monkeypatch.setenv("LTLF_GENERATION_MAX_TOKENS", "2048")
    monkeypatch.setenv("METHOD_SYNTHESIS_MAX_TOKENS", "4096")
    monkeypatch.setenv("PLANNING_TIMEOUT", "900")
    monkeypatch.setenv("LTLF_GENERATION_BASE_URL", "https://api.ltlf.example/v1")
    monkeypatch.setenv("METHOD_SYNTHESIS_BASE_URL", "https://api.method.example/v1")
    monkeypatch.setenv("LTLF_GENERATION_SESSION_ID", "ltlf-session")
    monkeypatch.setenv("METHOD_SYNTHESIS_SESSION_ID", "method-session")
    monkeypatch.setenv("EVALUATION_DOMAIN_SOURCE", "generated")

    config = Config()

    assert config.ltlf_generation_api_key == "sk-ltlf"
    assert config.method_synthesis_api_key == "sk-method"
    assert config.ltlf_generation_model == "deepseek/deepseek-chat-v3-0324"
    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"
    assert config.ltlf_generation_timeout == 120
    assert config.method_synthesis_timeout == 240
    assert config.ltlf_generation_max_tokens == 2048
    assert config.method_synthesis_max_tokens == 4096
    assert config.planning_timeout == 900
    assert config.ltlf_generation_base_url == "https://api.ltlf.example/v1"
    assert config.method_synthesis_base_url == "https://api.method.example/v1"
    assert config.ltlf_generation_session_id == "ltlf-session"
    assert config.method_synthesis_session_id == "method-session"
    assert config.evaluation_domain_source == "generated"


def test_method_synthesis_max_tokens_defaults_to_one_shot_library_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_MAX_TOKENS", raising=False)

    config = Config()

    assert config.method_synthesis_max_tokens == 48000


def test_ltlf_generation_timeout_defaults_to_kimi_long_reasoning_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LTLF_GENERATION_TIMEOUT", raising=False)

    config = Config()

    assert config.ltlf_generation_timeout == 1000


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


def test_ltlf_generation_model_defaults_to_pinned_kimi_when_no_env_is_set(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LTLF_GENERATION_MODEL", raising=False)

    config = Config()

    assert config.ltlf_generation_model == "moonshotai/kimi-k2.6"


def test_ltlf_generation_model_uses_stage_specific_override(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_MODEL", "provider/ltlf-model")

    config = Config()

    assert config.ltlf_generation_model == "provider/ltlf-model"


def test_method_synthesis_model_defaults_to_pinned_kimi_chat(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_MODEL", raising=False)

    config = Config()

    assert config.method_synthesis_model == "moonshotai/kimi-k2.6"


def test_method_synthesis_api_key_uses_method_specific_key(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.setenv("METHOD_SYNTHESIS_API_KEY", "sk-method")

    config = Config()

    assert config.method_synthesis_api_key == "sk-method"
    assert config.ltlf_generation_api_key == "sk-ltlf"


def test_method_synthesis_api_key_does_not_fall_back_to_ltlf_generation_key(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.delenv("METHOD_SYNTHESIS_API_KEY", raising=False)

    config = Config()

    assert config.method_synthesis_api_key is None


def test_dotenv_merge_preserves_explicit_shell_overrides(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_MODEL", "shell-ltlf-model")
    monkeypatch.setenv("METHOD_SYNTHESIS_MODEL", "shell-method-synthesis-model")

    config = Config()
    config._merge_env_lines(
        [
            "LTLF_GENERATION_MODEL=file-ltlf-model",
            "METHOD_SYNTHESIS_MODEL=file-method-synthesis-model",
        ]
    )

    assert config.ltlf_generation_model == "shell-ltlf-model"
    assert config.method_synthesis_model == "shell-method-synthesis-model"


def test_method_synthesis_model_uses_stage_specific_override(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_MODEL", raising=False)

    config = Config()
    config._merge_env_lines(
        [
            "METHOD_SYNTHESIS_MODEL=provider/method-model",
        ]
    )

    assert config.method_synthesis_model == "provider/method-model"


def test_generation_session_ids_default_to_distinct_values(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LTLF_GENERATION_SESSION_ID", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_SESSION_ID", raising=False)

    config = Config()

    assert config.ltlf_generation_session_id == "ltlf-generation"
    assert config.method_synthesis_session_id == "method-synthesis"
    assert config.ltlf_generation_session_id != config.method_synthesis_session_id


def test_evaluation_domain_source_defaults_to_benchmark(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("EVALUATION_DOMAIN_SOURCE", raising=False)

    config = Config()

    assert config.evaluation_domain_source == "benchmark"
