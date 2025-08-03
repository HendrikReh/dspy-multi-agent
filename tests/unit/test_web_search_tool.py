# tests/unit/test_web_search_tool.py
"""Unit tests for WebSearchTool."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from agents.researcher import WebSearchTool


@pytest.mark.unit
class TestWebSearchTool:
    """Test WebSearchTool functionality."""

    @pytest.mark.asyncio
    async def test_init_with_api_key(self):
        """Test WebSearchTool initialization with API key."""
        with patch('agents.researcher.AsyncTavilyClient') as mock_client:
            tool = WebSearchTool(api_key="test-key")
            assert tool.api_key == "test-key"
            mock_client.assert_called_once_with(api_key="test-key")
            assert tool.client is not None

    @pytest.mark.asyncio
    async def test_init_without_api_key_env_set(self, monkeypatch):
        """Test WebSearchTool initialization without API key but env var set."""
        monkeypatch.setenv("TAVILY_API_KEY", "env-test-key")
        with patch('agents.researcher.AsyncTavilyClient') as mock_client:
            tool = WebSearchTool()
            assert tool.api_key == "env-test-key"
            mock_client.assert_called_once_with(api_key="env-test-key")

    @pytest.mark.asyncio
    async def test_init_without_any_key(self, monkeypatch):
        """Test WebSearchTool initialization without any API key."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = WebSearchTool()
        assert tool.api_key is None
        assert tool.client is None

    @pytest.mark.asyncio
    async def test_search_with_valid_client(self, mock_tavily_client):
        """Test search functionality with valid client."""
        with patch('agents.researcher.AsyncTavilyClient', return_value=mock_tavily_client):
            tool = WebSearchTool(api_key="test-key")
            results = await tool.search("test query", num_results=5)
            
            assert len(results) == 3  # 2 results + 1 AI summary
            assert results[0]["title"] == "AI-Generated Summary"
            assert results[0]["snippet"] == "Test AI-generated summary"
            assert results[1]["title"] == "Test Result 1"
            assert results[2]["title"] == "Test Result 2"
            
            mock_tavily_client.search.assert_called_once_with(
                query="test query",
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                include_domains=[],
                exclude_domains=[]
            )

    @pytest.mark.asyncio
    async def test_search_without_client(self):
        """Test search functionality without client."""
        tool = WebSearchTool()
        tool.client = None
        results = await tool.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_exception(self, mock_tavily_client):
        """Test search functionality when exception occurs."""
        mock_tavily_client.search.side_effect = Exception("API Error")
        with patch('agents.researcher.AsyncTavilyClient', return_value=mock_tavily_client):
            tool = WebSearchTool(api_key="test-key")
            results = await tool.search("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_without_answer(self, mock_tavily_client):
        """Test search functionality when no AI answer is provided."""
        mock_tavily_client.search.return_value = {
            "results": [
                {
                    "title": "Test Result",
                    "content": "Test content",
                    "url": "https://example.com"
                }
            ],
            "answer": None
        }
        with patch('agents.researcher.AsyncTavilyClient', return_value=mock_tavily_client):
            tool = WebSearchTool(api_key="test-key")
            results = await tool.search("test query")
            assert len(results) == 1
            assert results[0]["title"] == "Test Result"

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test close method."""
        tool = WebSearchTool(api_key="test-key")
        await tool.close()  # Should not raise any exceptions