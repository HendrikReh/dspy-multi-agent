# src/signatures/agent_signatures.py
import dspy
from typing import List
from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    findings: List[str] = Field(description="Key research findings")
    sources: List[str] = Field(description="Source citations")
    confidence_score: float = Field(ge=0.0, le=1.0)


class ResearchSignature(dspy.Signature):
    """Research agent for gathering information on topics."""

    topic: str = dspy.InputField(description="Research topic")
    context: str = dspy.InputField(description="Additional context")
    research_results: str = dspy.OutputField(
        description="Comprehensive research findings"
    )
    key_points: List[str] = dspy.OutputField(description="Main points discovered")
    sources: List[str] = dspy.OutputField(description="Relevant sources")


class WritingSignature(dspy.Signature):
    """Writer agent for creating content based on research."""

    topic: str = dspy.InputField(description="Writing topic")
    research_data: str = dspy.InputField(description="Research findings")
    key_points: List[str] = dspy.InputField(description="Key points to cover")
    target_audience: str = dspy.InputField(description="Target audience")
    article: str = dspy.OutputField(description="Well-structured article")
    summary: str = dspy.OutputField(description="Article summary")
