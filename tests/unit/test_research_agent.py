# tests/unit/test_research_agent.py
"""Unit tests for ResearchAgent."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import dspy
from agents.researcher import ResearchAgent, WebSearchTool


@pytest.mark.unit
class TestResearchAgent:
    """Test ResearchAgent functionality."""

    @pytest.fixture
    def mock_search_tool(self):
        """Create a mock search tool."""
        tool = Mock(spec=WebSearchTool)
        tool.search = AsyncMock(return_value=[
            {
                "title": "Research Result 1",
                "snippet": "Important research finding 1",
                "url": "https://example.com/1"
            },
            {
                "title": "Research Result 2",
                "snippet": "Important research finding 2",
                "url": "https://example.com/2"
            }
        ])
        tool.close = AsyncMock()
        return tool

    @pytest.fixture
    def mock_dspy_chain(self):
        """Create a mock DSPy ChainOfThought."""
        mock_result = Mock()
        mock_result.research_results = "Comprehensive research on the topic"
        mock_result.key_points = ["Key point 1", "Key point 2", "Key point 3"]
        mock_result.sources = ["https://example.com/1", "https://example.com/2"]
        
        mock_chain = Mock()
        mock_chain.return_value = mock_result
        return mock_chain

    def test_init_with_search_tool(self, mock_search_tool):
        """Test ResearchAgent initialization with search tool."""
        agent = ResearchAgent(search_tool=mock_search_tool)
        assert agent.search_tool == mock_search_tool
        assert agent._owns_search_tool is False

    def test_init_without_search_tool(self):
        """Test ResearchAgent initialization without search tool."""
        with patch('agents.researcher.WebSearchTool') as mock_tool_class:
            agent = ResearchAgent()
            assert agent._owns_search_tool is True
            mock_tool_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_with_search_results(self, mock_search_tool, mock_dspy_chain):
        """Test forward method with search results."""
        with patch('dspy.ChainOfThought', return_value=mock_dspy_chain):
            agent = ResearchAgent(search_tool=mock_search_tool)
            
            result = await agent.forward(
                topic="AI in healthcare",
                context="Medical context",
                max_sources=5
            )
            
            # Verify search was called
            mock_search_tool.search.assert_called_once_with("AI in healthcare", 5)
            
            # Verify DSPy chain was called with proper context
            mock_dspy_chain.assert_called_once()
            call_args = mock_dspy_chain.call_args[1]
            assert "AI in healthcare" in call_args["topic"]
            assert "Medical context" in call_args["context"]
            assert "Research Result 1" in call_args["context"]
            assert "Research Result 2" in call_args["context"]
            
            # Verify result structure
            assert result["topic"] == "AI in healthcare"
            assert result["research_results"] == "Comprehensive research on the topic"
            assert len(result["key_points"]) == 3
            assert len(result["sources"]) == 2
            assert len(result["search_results"]) == 2

    @pytest.mark.asyncio
    async def test_forward_without_context(self, mock_search_tool, mock_dspy_chain):
        """Test forward method without initial context."""
        with patch('dspy.ChainOfThought', return_value=mock_dspy_chain):
            agent = ResearchAgent(search_tool=mock_search_tool)
            
            result = await agent.forward(topic="Climate change")
            
            # Verify default values
            assert result["topic"] == "Climate change"
            mock_search_tool.search.assert_called_once_with("Climate change", 5)

    @pytest.mark.asyncio
    async def test_forward_with_empty_search_results(self, mock_dspy_chain):
        """Test forward method when search returns no results."""
        mock_search_tool = Mock(spec=WebSearchTool)
        mock_search_tool.search = AsyncMock(return_value=[])
        
        with patch('dspy.ChainOfThought', return_value=mock_dspy_chain):
            agent = ResearchAgent(search_tool=mock_search_tool)
            
            result = await agent.forward(topic="Obscure topic")
            
            # Should still call DSPy chain even with no search results
            mock_dspy_chain.assert_called_once()
            assert result["search_results"] == []

    @pytest.mark.asyncio
    async def test_close_owns_search_tool(self, mock_search_tool):
        """Test close method when agent owns search tool."""
        with patch('agents.researcher.WebSearchTool', return_value=mock_search_tool):
            agent = ResearchAgent()  # Creates its own search tool
            await agent.close()
            mock_search_tool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_not_own_search_tool(self, mock_search_tool):
        """Test close method when agent doesn't own search tool."""
        agent = ResearchAgent(search_tool=mock_search_tool)
        await agent.close()
        mock_search_tool.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_context_formatting(self, mock_search_tool, mock_dspy_chain):
        """Test that search results are properly formatted in context."""
        with patch('dspy.ChainOfThought', return_value=mock_dspy_chain):
            agent = ResearchAgent(search_tool=mock_search_tool)
            
            await agent.forward(topic="Test topic", context="Initial context")
            
            # Check that context was properly formatted
            call_args = mock_dspy_chain.call_args[1]
            context = call_args["context"]
            assert "Initial context" in context
            assert "Web Search Results:" in context
            assert "Source: Research Result 1" in context
            assert "Important research finding 1" in context
            assert "URL: https://example.com/1" in context