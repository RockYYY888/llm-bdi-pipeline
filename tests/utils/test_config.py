"""
Configuration tests for shared and stage-specific language-model settings.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import Config


def test_stage_specific_generation_config_reads_expected_fields(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LANGUAGE_MODEL_API_KEY", "sk-shared")
    monkeypatch.setenv("LANGUAGE_MODEL_MODEL", "provider/shared-model")
    monkeypatch.setenv("LANGUAGE_MODEL_BASE_URL", "https://api.shared.example/v1")
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.setenv("METHOD_SYNTHESIS_API_KEY", "sk-method")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_API_KEY", "sk-direct")
    monkeypatch.setenv("LTLF_GENERATION_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("METHOD_SYNTHESIS_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("LTLF_GENERATION_TIMEOUT", "120")
    monkeypatch.setenv("METHOD_SYNTHESIS_TIMEOUT", "240")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_TIMEOUT", "360")
    monkeypatch.setenv("LTLF_GENERATION_MAX_TOKENS", "2048")
    monkeypatch.setenv("METHOD_SYNTHESIS_MAX_TOKENS", "4096")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_MAX_TOKENS", "8192")
    monkeypatch.setenv("PLANNING_TIMEOUT", "900")
    monkeypatch.setenv("LTLF_GENERATION_BASE_URL", "https://api.ltlf.example/v1")
    monkeypatch.setenv("METHOD_SYNTHESIS_BASE_URL", "https://api.method.example/v1")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_BASE_URL", "https://api.direct.example/v1")
    monkeypatch.setenv("LTLF_GENERATION_SESSION_ID", "ltlf-session")
    monkeypatch.setenv("METHOD_SYNTHESIS_SESSION_ID", "method-session")
    monkeypatch.setenv("DIRECT_PLAN_GENERATION_SESSION_ID", "direct-session")
    monkeypatch.setenv("EVALUATION_DOMAIN_SOURCE", "generated")

    config = Config()

    assert config.language_model_api_key == "sk-shared"
    assert config.language_model_model == "provider/shared-model"
    assert config.language_model_base_url == "https://api.shared.example/v1"
    assert config.ltlf_generation_api_key == "sk-ltlf"
    assert config.method_synthesis_api_key == "sk-method"
    assert config.direct_plan_generation_api_key == "sk-direct"
    assert config.ltlf_generation_model == "deepseek-v4-pro"
    assert config.method_synthesis_model == "deepseek-v4-pro"
    assert config.direct_plan_generation_model == "deepseek-v4-pro"
    assert config.ltlf_generation_timeout == 120
    assert config.method_synthesis_timeout == 240
    assert config.direct_plan_generation_timeout == 360
    assert config.ltlf_generation_max_tokens == 2048
    assert config.method_synthesis_max_tokens == 4096
    assert config.direct_plan_generation_max_tokens == 8192
    assert config.planning_timeout == 900
    assert config.ltlf_generation_base_url == "https://api.ltlf.example/v1"
    assert config.method_synthesis_base_url == "https://api.method.example/v1"
    assert config.direct_plan_generation_base_url == "https://api.direct.example/v1"
    assert config.ltlf_generation_session_id == "ltlf-session"
    assert config.method_synthesis_session_id == "method-session"
    assert config.direct_plan_generation_session_id == "direct-session"
    assert config.evaluation_domain_source == "generated"


def test_method_synthesis_max_tokens_defaults_to_one_shot_library_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_MAX_TOKENS", raising=False)

    config = Config()

    assert config.method_synthesis_max_tokens == 144000


def test_ltlf_generation_timeout_defaults_to_long_generation_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LTLF_GENERATION_TIMEOUT", raising=False)

    config = Config()

    assert config.ltlf_generation_timeout == 1000


def test_method_synthesis_timeout_defaults_to_longer_one_shot_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("METHOD_SYNTHESIS_TIMEOUT", raising=False)

    config = Config()

    assert config.method_synthesis_timeout == 2400


def test_direct_plan_generation_timeout_defaults_to_evaluation_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_TIMEOUT", raising=False)

    config = Config()

    assert config.direct_plan_generation_timeout == 1800


def test_planning_timeout_defaults_to_large_runtime_budget(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("PLANNING_TIMEOUT", raising=False)

    config = Config()

    assert config.planning_timeout == 600


def test_ltlf_generation_model_defaults_to_deepseek_v4_pro_when_no_env_is_set(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LANGUAGE_MODEL_MODEL", raising=False)
    monkeypatch.delenv("LTLF_GENERATION_MODEL", raising=False)

    config = Config()

    assert config.ltlf_generation_model == "deepseek-v4-pro"


def test_generation_stages_fall_back_to_shared_language_model_config(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LANGUAGE_MODEL_API_KEY", "sk-shared")
    monkeypatch.setenv("LANGUAGE_MODEL_MODEL", "provider/shared-model")
    monkeypatch.setenv("LANGUAGE_MODEL_BASE_URL", "https://api.shared.example/v1")
    monkeypatch.delenv("LTLF_GENERATION_API_KEY", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_API_KEY", raising=False)
    monkeypatch.delenv("LTLF_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("LTLF_GENERATION_BASE_URL", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_BASE_URL", raising=False)

    config = Config()

    assert config.ltlf_generation_api_key == "sk-shared"
    assert config.method_synthesis_api_key == "sk-shared"
    assert config.direct_plan_generation_api_key == "sk-shared"
    assert config.ltlf_generation_model == "provider/shared-model"
    assert config.method_synthesis_model == "provider/shared-model"
    assert config.direct_plan_generation_model == "provider/shared-model"
    assert config.ltlf_generation_base_url == "https://api.shared.example/v1"
    assert config.method_synthesis_base_url == "https://api.shared.example/v1"
    assert config.direct_plan_generation_base_url == "https://api.shared.example/v1"


def test_ltlf_generation_model_uses_stage_specific_override(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LTLF_GENERATION_MODEL", "provider/ltlf-model")

    config = Config()

    assert config.ltlf_generation_model == "provider/ltlf-model"


def test_method_synthesis_model_defaults_to_deepseek_v4_pro(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LANGUAGE_MODEL_MODEL", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_MODEL", raising=False)

    config = Config()

    assert config.method_synthesis_model == "deepseek-v4-pro"


def test_direct_plan_generation_model_defaults_to_deepseek_v4_pro(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LANGUAGE_MODEL_MODEL", raising=False)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_MODEL", raising=False)

    config = Config()

    assert config.direct_plan_generation_model == "deepseek-v4-pro"


def test_generation_base_urls_default_to_deepseek_openai_endpoint(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("LANGUAGE_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("LTLF_GENERATION_BASE_URL", raising=False)
    monkeypatch.delenv("METHOD_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_BASE_URL", raising=False)

    config = Config()

    assert config.ltlf_generation_base_url == "https://api.deepseek.com"
    assert config.method_synthesis_base_url == "https://api.deepseek.com"
    assert config.direct_plan_generation_base_url == "https://api.deepseek.com"


def test_method_synthesis_api_key_uses_method_specific_key_before_shared_key(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LANGUAGE_MODEL_API_KEY", "sk-shared")
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.setenv("METHOD_SYNTHESIS_API_KEY", "sk-method")

    config = Config()

    assert config.method_synthesis_api_key == "sk-method"
    assert config.ltlf_generation_api_key == "sk-ltlf"


def test_method_synthesis_api_key_falls_back_to_shared_key_not_ltlf_key(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LANGUAGE_MODEL_API_KEY", "sk-shared")
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.delenv("METHOD_SYNTHESIS_API_KEY", raising=False)

    config = Config()

    assert config.method_synthesis_api_key == "sk-shared"


def test_direct_plan_generation_api_key_falls_back_to_shared_key_not_other_stage_keys(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.setenv("LANGUAGE_MODEL_API_KEY", "sk-shared")
    monkeypatch.setenv("LTLF_GENERATION_API_KEY", "sk-ltlf")
    monkeypatch.setenv("METHOD_SYNTHESIS_API_KEY", "sk-method")
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_API_KEY", raising=False)

    config = Config()

    assert config.direct_plan_generation_api_key == "sk-shared"


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
    monkeypatch.delenv("DIRECT_PLAN_GENERATION_SESSION_ID", raising=False)

    config = Config()

    assert config.ltlf_generation_session_id == "ltlf-generation"
    assert config.method_synthesis_session_id == "method-synthesis"
    assert config.direct_plan_generation_session_id == "direct-plan-generation"
    assert config.ltlf_generation_session_id != config.method_synthesis_session_id
    assert config.direct_plan_generation_session_id != config.method_synthesis_session_id


def test_evaluation_domain_source_defaults_to_benchmark(monkeypatch):
    monkeypatch.setattr(Config, "_load_env", lambda self: None)
    monkeypatch.delenv("EVALUATION_DOMAIN_SOURCE", raising=False)

    config = Config()

    assert config.evaluation_domain_source == "benchmark"
