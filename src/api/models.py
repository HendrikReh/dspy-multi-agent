# src/api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AgentRequest(BaseModel):
    """Request model for multi-agent processing."""

    query: str = Field(..., description="Research and writing request")
    target_audience: str = Field(default="general", description="Target audience")
    max_sources: int = Field(default=5, ge=1, le=20, description="Maximum sources")
    style: str = Field(default="informative", description="Writing style")


class AgentResponse(BaseModel):
    """Response model for multi-agent processing."""

    status: str = Field(description="Processing status")
    topic: str = Field(description="Extracted topic")
    article: str = Field(description="Generated article")
    summary: str = Field(description="Article summary")
    sources: List[str] = Field(description="Research sources")
    key_points: List[str] = Field(description="Key points covered")
    processing_time: float = Field(description="Processing time in seconds")
    agent_id: str = Field(description="Agent instance ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
