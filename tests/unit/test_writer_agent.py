# tests/unit/test_writer_agent.py
"""Unit tests for WriterAgent."""
import pytest
from unittest.mock import Mock, patch
import dspy
from agents.writer import WriterAgent


@pytest.mark.unit
class TestWriterAgent:
    """Test WriterAgent functionality."""

    @pytest.fixture
    def mock_writing_signature(self):
        """Create a mock writing signature result."""
        mock_result = Mock()
        mock_result.article = "This is a draft article about the topic."
        mock_result.summary = "This is a concise summary of the article."
        return mock_result

    @pytest.fixture
    def mock_editor_result(self):
        """Create a mock editor result."""
        mock_result = Mock()
        mock_result.polished_article = "This is a polished and refined article."
        return mock_result

    def test_init(self):
        """Test WriterAgent initialization."""
        with patch('dspy.ChainOfThought') as mock_cot, \
             patch('dspy.Predict') as mock_predict:
            agent = WriterAgent()
            # Should create 1 ChainOfThought and 1 Predict
            mock_cot.assert_called_once()
            mock_predict.assert_called_once_with("draft -> polished_article")

    def test_forward_complete_flow(self, mock_writing_signature, mock_editor_result):
        """Test forward method with complete flow."""
        # Mock the ChainOfThought and Predict instances
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            result = agent.forward(
                topic="AI in Healthcare",
                research_data="Research findings about AI applications",
                key_points=["Point 1", "Point 2"],
                target_audience="Healthcare professionals"
            )
            
            # Verify writer was called
            mock_writer_chain.assert_called_once()
            writer_call = mock_writer_chain.call_args[1]
            assert writer_call["topic"] == "AI in Healthcare"
            assert writer_call["research_data"] == "Research findings about AI applications"
            assert writer_call["target_audience"] == "Healthcare professionals"
            assert writer_call["key_points"] == ["Point 1", "Point 2"]
            
            # Verify editor was called
            mock_editor.assert_called_once()
            editor_call = mock_editor.call_args[1]
            assert editor_call["draft"] == "This is a draft article about the topic."
            
            # Verify result structure
            assert result["topic"] == "AI in Healthcare"
            assert result["article"] == "This is a polished and refined article."
            assert result["summary"] == "This is a concise summary of the article."
            assert result["target_audience"] == "Healthcare professionals"
            assert result["key_points_covered"] == ["Point 1", "Point 2"]

    def test_forward_with_default_audience(self, mock_writing_signature, mock_editor_result):
        """Test forward method with default target audience."""
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            result = agent.forward(
                topic="Climate Change",
                research_data="Climate research",
                key_points=["Point 1"]
                # No target_audience specified
            )
            
            # Should use default audience
            writer_call = mock_writer_chain.call_args[1]
            assert writer_call["target_audience"] == "general"
            assert result["target_audience"] == "general"

    def test_forward_with_empty_inputs(self, mock_writing_signature, mock_editor_result):
        """Test forward method with empty inputs."""
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            result = agent.forward(
                topic="",
                research_data="",
                key_points=[]
            )
            
            # Should still process even with empty inputs
            assert result["article"] == "This is a polished and refined article."
            assert result["summary"] == "This is a concise summary of the article."
            assert result["key_points_covered"] == []

    def test_forward_preserves_key_points(self, mock_writing_signature, mock_editor_result):
        """Test that forward method preserves key points list."""
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            test_points = ["Point A", "Point B", "Point C"]
            result = agent.forward(
                topic="Test Topic",
                research_data="Test results",
                key_points=test_points
            )
            
            # Key points should be preserved exactly
            assert result["key_points_covered"] == test_points

    def test_forward_with_long_content(self, mock_writing_signature, mock_editor_result):
        """Test forward method with long content."""
        # Create long content
        long_research = "Very long research results. " * 100
        many_points = [f"Point {i}" for i in range(50)]
        
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            result = agent.forward(
                topic="Complex Topic",
                research_data=long_research,
                key_points=many_points
            )
            
            # Should handle long content without issues
            assert result["article"] is not None
            assert result["summary"] is not None
            assert len(result["key_points_covered"]) == 50

    def test_forward_with_custom_style(self, mock_writing_signature, mock_editor_result):
        """Test forward method with custom style parameter."""
        mock_writer_chain = Mock(return_value=mock_writing_signature)
        mock_editor = Mock(return_value=mock_editor_result)
        
        with patch('dspy.ChainOfThought', return_value=mock_writer_chain), \
             patch('dspy.Predict', return_value=mock_editor):
            agent = WriterAgent()
            
            # Note: style parameter is accepted but not used in current implementation
            result = agent.forward(
                topic="Test Topic",
                research_data="Test data",
                key_points=["Point 1"],
                style="academic"  # This parameter is ignored
            )
            
            # Should still work even though style is not used
            assert result["article"] is not None