# tests/unit/test_config.py
"""Unit tests for Config."""
import pytest
import os
from unittest.mock import patch
from utils.config import Config


@pytest.mark.unit
class TestConfig:
    """Test Config functionality."""

    def test_load_environment_variables(self, monkeypatch):
        """Test loading environment variables."""
        # Set test environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
        monkeypatch.setenv("MODEL_NAME", "gpt-4")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        monkeypatch.setenv("MAX_TOKENS", "1500")
        monkeypatch.setenv("ASYNC_WORKERS", "8")
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        config = Config()
        
        assert config.openai_api_key == "test-openai-key"
        assert config.tavily_api_key == "test-tavily-key"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1500
        assert config.async_workers == 8
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 9000
        assert config.log_level == "DEBUG"

    def test_default_values(self, monkeypatch):
        """Test default values when environment variables are not set."""
        # Only set required environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Clear optional environment variables
        for var in ["MODEL_NAME", "TEMPERATURE", "MAX_TOKENS", "ASYNC_WORKERS", 
                   "API_HOST", "API_PORT", "LOG_LEVEL", "TAVILY_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        
        config = Config()
        
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.async_workers == 4
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.log_level == "INFO"
        assert config.tavily_api_key is None

    def test_missing_openai_key(self, monkeypatch):
        """Test error when OpenAI API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
            Config()

    def test_invalid_temperature_too_low(self, monkeypatch):
        """Test error when temperature is too low."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("TEMPERATURE", "-0.1")
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            Config()

    def test_invalid_temperature_too_high(self, monkeypatch):
        """Test error when temperature is too high."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("TEMPERATURE", "2.1")
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            Config()

    def test_boundary_temperature_values(self, monkeypatch):
        """Test boundary values for temperature."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        # Test minimum valid temperature
        monkeypatch.setenv("TEMPERATURE", "0")
        config = Config()
        assert config.temperature == 0.0
        
        # Test maximum valid temperature
        monkeypatch.setenv("TEMPERATURE", "2")
        config = Config()
        assert config.temperature == 2.0

    def test_numeric_conversion_errors(self, monkeypatch):
        """Test errors in numeric conversions."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        # Invalid temperature format
        monkeypatch.setenv("TEMPERATURE", "not-a-number")
        with pytest.raises(ValueError):
            Config()
        
        # Invalid max_tokens format
        monkeypatch.setenv("TEMPERATURE", "0.7")  # Reset to valid
        monkeypatch.setenv("MAX_TOKENS", "not-a-number")
        with pytest.raises(ValueError):
            Config()
        
        # Invalid port format
        monkeypatch.setenv("MAX_TOKENS", "2000")  # Reset to valid
        monkeypatch.setenv("API_PORT", "not-a-number")
        with pytest.raises(ValueError):
            Config()

    def test_warning_for_missing_tavily_key(self, monkeypatch, capsys):
        """Test warning when Tavily API key is not configured."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        
        config = Config()
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning: TAVILY_API_KEY not configured" in captured.out
        assert config.tavily_api_key is None

    def test_no_warning_when_tavily_key_present(self, monkeypatch, capsys):
        """Test no warning when Tavily API key is configured."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
        
        config = Config()
        
        # Check that no warning was printed
        captured = capsys.readouterr()
        assert "Warning: TAVILY_API_KEY not configured" not in captured.out
        assert config.tavily_api_key == "test-tavily-key"