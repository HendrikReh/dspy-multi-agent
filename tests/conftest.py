# tests/conftest.py
"""Pytest configuration and fixtures."""
import os
import sys
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_tavily_client():
    """Mock Tavily client for testing."""
    client = AsyncMock()
    client.search = AsyncMock(return_value={
        "results": [
            {
                "title": "Test Result 1",
                "content": "Test content for result 1",
                "url": "https://example.com/1"
            },
            {
                "title": "Test Result 2", 
                "content": "Test content for result 2",
                "url": "https://example.com/2"
            }
        ],
        "answer": "Test AI-generated summary"
    })
    return client


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model."""
    lm = Mock()
    lm.forward = Mock(return_value={
        "research_results": "Test research results",
        "key_points": ["Point 1", "Point 2"],
        "sources": ["Source 1", "Source 2"],
        "draft_article": "Test draft article",
        "polished_article": "Test polished article"
    })
    return lm


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "openai_api_key": "test-api-key",
        "tavily_api_key": "test-tavily-key",
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 2000,
        "async_workers": 4
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    monkeypatch.setenv("MAX_TOKENS", "2000")