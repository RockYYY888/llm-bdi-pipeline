"""
Unit tests for Configuration Module

Tests cover:
- Configuration loading from environment
- API key validation
- Model configuration
- Default values
"""

import pytest
import os
from unittest.mock import patch
from config import Config, get_config


# ===== Test Config Class =====

class TestConfig:
    """Test pipeline configuration"""

    def test_config_with_api_key(self, mock_env_with_api_key):
        """Test configuration loads API key from environment"""
        config = Config()
        assert config.openai_api_key == "sk-test-mock-key-12345"
        assert config.openai_model == "gpt-4o-mini"

    def test_config_without_api_key(self, mock_env_no_api_key):
        """Test configuration handles missing API key"""
        config = Config()
        assert config.openai_api_key is None

    def test_config_default_model(self, mock_env_no_api_key):
        """Test configuration uses default model"""
        config = Config()
        assert config.openai_model == "gpt-4o-mini"

    def test_config_custom_model(self, monkeypatch, mock_env_no_api_key):
        """Test configuration uses custom model from environment"""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        config = Config()
        assert config.openai_model == "gpt-4o"

    def test_config_base_url(self, monkeypatch, mock_env_no_api_key):
        """Test configuration loads custom base URL"""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api.url")
        config = Config()
        assert config.openai_base_url == "https://custom.api.url"

    def test_config_no_base_url(self, mock_env_no_api_key):
        """Test configuration handles missing base URL"""
        config = Config()
        assert config.openai_base_url is None

    def test_validate_with_api_key(self, mock_env_with_api_key):
        """Test validation passes with API key"""
        config = Config()
        assert config.validate() is True

    def test_validate_without_api_key(self, mock_env_no_api_key):
        """Test validation fails without API key"""
        config = Config()
        assert config.validate() is False

    def test_use_llm_planner_default(self, mock_env_no_api_key):
        """Test USE_LLM_PLANNER defaults to False"""
        config = Config()
        assert config.use_llm_planner is False

    def test_use_llm_planner_enabled(self, monkeypatch):
        """Test USE_LLM_PLANNER can be enabled"""
        monkeypatch.setenv("USE_LLM_PLANNER", "true")
        config = Config()
        assert config.use_llm_planner is True

    def test_use_llm_planner_case_insensitive(self, monkeypatch):
        """Test USE_LLM_PLANNER is case insensitive"""
        monkeypatch.setenv("USE_LLM_PLANNER", "TRUE")
        config = Config()
        assert config.use_llm_planner is True

    def test_use_llm_planner_false_values(self, monkeypatch):
        """Test USE_LLM_PLANNER correctly interprets false values"""
        test_cases = ["false", "False", "FALSE", "0", "no"]
        for value in test_cases:
            monkeypatch.setenv("USE_LLM_PLANNER", value)
            config = Config()
            assert config.use_llm_planner is False


# ===== Test get_config Function =====

class TestGetConfig:
    """Test get_config singleton function"""

    def test_get_config_returns_instance(self, mock_env_with_api_key):
        """Test get_config returns Config instance"""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_singleton(self, mock_env_with_api_key):
        """Test get_config returns same instance (singleton pattern)"""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_get_config_with_different_environments(self, monkeypatch):
        """Test get_config adapts to environment changes"""
        # Note: This requires clearing the singleton cache if implemented
        monkeypatch.setenv("OPENAI_API_KEY", "sk-first-key")
        config1 = get_config()
        assert config1.openai_api_key == "sk-first-key"


# ===== Test Environment Variable Handling =====

class TestEnvironmentVariables:
    """Test environment variable loading and defaults"""

    def test_missing_all_env_vars(self, mock_env_no_api_key):
        """Test configuration with all environment variables missing"""
        config = Config()
        assert config.openai_api_key is None
        assert config.openai_model == "gpt-4o-mini"  # Default
        assert config.openai_base_url is None
        assert config.use_llm_planner is False  # Default

    def test_partial_env_vars(self, monkeypatch, mock_env_no_api_key):
        """Test configuration with partial environment variables"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-partial-test")

        config = Config()
        assert config.openai_api_key == "sk-partial-test"
        assert config.openai_model == "gpt-4o-mini"  # Default
        assert config.openai_base_url is None

    def test_all_env_vars_set(self, monkeypatch, mock_env_no_api_key):
        """Test configuration with all environment variables set"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-full-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.url")
        monkeypatch.setenv("USE_LLM_PLANNER", "true")

        config = Config()
        assert config.openai_api_key == "sk-full-test"
        assert config.openai_model == "gpt-4"
        assert config.openai_base_url == "https://test.url"
        assert config.use_llm_planner is True


# ===== Test Edge Cases =====

class TestConfigEdgeCases:
    """Test configuration edge cases and error handling"""

    def test_empty_api_key(self, monkeypatch, mock_env_no_api_key):
        """Test configuration with empty API key string"""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        config = Config()
        # Empty string is not None, but validate should fail
        assert config.openai_api_key == ""
        assert config.validate() is False

    def test_whitespace_api_key(self, monkeypatch, mock_env_no_api_key):
        """Test configuration with whitespace API key"""
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        config = Config()
        assert config.validate() is False

    def test_api_key_with_spaces(self, monkeypatch, mock_env_no_api_key):
        """Test configuration preserves API key with internal spaces"""
        key_with_spaces = "sk-test key with spaces"
        monkeypatch.setenv("OPENAI_API_KEY", key_with_spaces)
        config = Config()
        assert config.openai_api_key == key_with_spaces

    def test_model_name_variations(self, monkeypatch, mock_env_no_api_key):
        """Test configuration handles various model name formats"""
        model_names = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "custom-model-name"
        ]
        for model_name in model_names:
            monkeypatch.setenv("OPENAI_MODEL", model_name)
            config = Config()
            assert config.openai_model == model_name
