# tests/unit/test_coordinator.py
"""Unit tests for MultiAgentCoordinator."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from agents.coordinator import MultiAgentCoordinator


@pytest.mark.unit  
class TestMultiAgentCoordinator:
    """Test MultiAgentCoordinator functionality."""

    @pytest.fixture
    def mock_research_agent(self):
        """Create a mock research agent."""
        agent = Mock()
        agent.forward = AsyncMock(return_value={
            "topic": "Test Topic",
            "research_results": "Detailed research findings",
            "key_points": ["Key point 1", "Key point 2", "Key point 3"],
            "sources": ["https://source1.com", "https://source2.com"],
            "search_results": [
                {"title": "Result 1", "snippet": "Content 1", "url": "https://source1.com"},
                {"title": "Result 2", "snippet": "Content 2", "url": "https://source2.com"}
            ],
            "context": "Full context with search results"
        })
        agent.close = AsyncMock()
        return agent

    @pytest.fixture
    def mock_writer_agent(self):
        """Create a mock writer agent."""
        agent = Mock()
        agent.forward = Mock(return_value={
            "topic": "Test Topic",
            "article": "Polished article content with proper formatting",
            "summary": "Concise summary of the article",
            "target_audience": "general audience",
            "key_points_covered": ["Key point 1", "Key point 2", "Key point 3"]
        })
        return agent

    @pytest.fixture
    def mock_task_planner(self):
        """Create a mock task planner."""
        planner = Mock()
        planner.return_value = Mock(research_plan="Research AI in healthcare", writing_plan="Write comprehensive article")
        return planner

    def test_init_with_api_key(self):
        """Test coordinator initialization with API key."""
        with patch('agents.coordinator.WebSearchTool') as mock_tool, \
             patch('agents.coordinator.ResearchAgent') as mock_research, \
             patch('agents.coordinator.WriterAgent') as mock_writer, \
             patch('dspy.Predict') as mock_predict:
            
            coordinator = MultiAgentCoordinator(tavily_api_key="test-key")
            
            # Verify WebSearchTool was created with API key
            mock_tool.assert_called_once_with("test-key")
            # Verify ResearchAgent was created with search tool
            mock_research.assert_called_once()
            # Verify WriterAgent was created
            mock_writer.assert_called_once()
            # Verify task planner was created
            mock_predict.assert_called_once_with("request -> research_plan, writing_plan")

    def test_init_without_api_key(self):
        """Test coordinator initialization without API key."""
        with patch('agents.coordinator.WebSearchTool') as mock_tool, \
             patch('agents.coordinator.ResearchAgent') as mock_research, \
             patch('agents.coordinator.WriterAgent') as mock_writer, \
             patch('dspy.Predict') as mock_predict:
            
            coordinator = MultiAgentCoordinator()
            
            # WebSearchTool should not be created
            mock_tool.assert_not_called()
            # ResearchAgent should be created with None
            mock_research.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_forward_complete_flow(self, mock_research_agent, mock_writer_agent, mock_task_planner):
        """Test forward method with complete flow."""
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('agents.coordinator.WriterAgent', return_value=mock_writer_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            
            coordinator = MultiAgentCoordinator(tavily_api_key="test-key")
            
            result = await coordinator.forward(
                request="Write about AI in healthcare",
                target_audience="medical professionals",
                max_sources=10
            )
            
            # Verify task planner was called
            mock_task_planner.assert_called_once_with(request="Write about AI in healthcare")
            
            # Verify research agent was called
            mock_research_agent.forward.assert_called_once_with(
                topic="Write about AI in healthcare",
                max_sources=10
            )
            
            # Verify writer agent was called
            mock_writer_agent.forward.assert_called_once_with(
                topic="Write about AI in healthcare",
                research_data="Detailed research findings",
                key_points=["Key point 1", "Key point 2", "Key point 3"],
                target_audience="medical professionals"
            )
            
            # Verify result structure
            assert result["request"] == "Write about AI in healthcare"
            assert result["topic"] == "Write about AI in healthcare"
            assert result["final_article"] == "Polished article content with proper formatting"
            assert result["summary"] == "Concise summary of the article"
            assert result["sources"] == ["https://source1.com", "https://source2.com"]
            assert "research_phase" in result
            assert "writing_phase" in result

    @pytest.mark.asyncio
    async def test_forward_with_defaults(self, mock_research_agent, mock_writer_agent, mock_task_planner):
        """Test forward method with default parameters."""
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('agents.coordinator.WriterAgent', return_value=mock_writer_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            
            coordinator = MultiAgentCoordinator()
            
            result = await coordinator.forward(request="Simple request")
            
            # Should use default values
            mock_research_agent.forward.assert_called_once_with(
                topic="Simple request",
                max_sources=5  # Default value
            )
            
            mock_writer_agent.forward.assert_called_once()
            writer_call = mock_writer_agent.forward.call_args[1]
            assert writer_call["target_audience"] == "general"  # Default value

    @pytest.mark.asyncio
    async def test_forward_error_in_research(self, mock_task_planner):
        """Test forward method error handling in research phase."""
        mock_research_agent = Mock()
        mock_research_agent.forward = AsyncMock(side_effect=Exception("Research failed"))
        
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            coordinator = MultiAgentCoordinator()
            
            with pytest.raises(Exception, match="Research failed"):
                await coordinator.forward(request="Test request")

    @pytest.mark.asyncio
    async def test_forward_error_in_writing(self, mock_research_agent, mock_task_planner):
        """Test forward method error handling in writing phase."""
        mock_writer_agent = Mock()
        mock_writer_agent.forward = Mock(side_effect=Exception("Writing failed"))
        
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('agents.coordinator.WriterAgent', return_value=mock_writer_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            coordinator = MultiAgentCoordinator()
            
            with pytest.raises(Exception, match="Writing failed"):
                await coordinator.forward(request="Test request")

    @pytest.mark.asyncio
    async def test_close_method(self, mock_research_agent):
        """Test close method."""
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('dspy.Predict'):
            coordinator = MultiAgentCoordinator(tavily_api_key="test-key")
            
            await coordinator.close()
            
            # Should close research agent
            mock_research_agent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_preserves_all_data(self, mock_research_agent, mock_writer_agent, mock_task_planner):
        """Test that forward preserves all data from agents."""
        # Add extra fields to mock responses
        mock_research_agent.forward.return_value["extra_research_field"] = "extra_value"
        mock_writer_agent.forward.return_value["extra_writer_field"] = "extra_value"
        
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('agents.coordinator.WriterAgent', return_value=mock_writer_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            
            coordinator = MultiAgentCoordinator()
            
            result = await coordinator.forward(request="Test request")
            
            # All original fields should be preserved
            assert result["research_phase"]["extra_research_field"] == "extra_value"
            assert result["writing_phase"]["extra_writer_field"] == "extra_value"

    @pytest.mark.asyncio
    async def test_task_planner_integration(self, mock_research_agent, mock_writer_agent):
        """Test task planner integration."""
        mock_plan = Mock(research_plan="Detailed research plan", writing_plan="Detailed writing plan")
        mock_task_planner = Mock(return_value=mock_plan)
        
        with patch('agents.coordinator.ResearchAgent', return_value=mock_research_agent), \
             patch('agents.coordinator.WriterAgent', return_value=mock_writer_agent), \
             patch('dspy.Predict', return_value=mock_task_planner):
            
            coordinator = MultiAgentCoordinator()
            
            result = await coordinator.forward(request="Complex request")
            
            # Task planner should have been called
            mock_task_planner.assert_called_once_with(request="Complex request")