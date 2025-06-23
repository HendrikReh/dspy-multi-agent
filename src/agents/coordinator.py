# src/agents/coordinator.py
import dspy
import asyncio
from typing import Dict, Any, Optional
from agents.researcher import ResearchAgent, WebSearchTool
from agents.writer import WriterAgent


class MultiAgentCoordinator(dspy.Module):
    """Coordinates researcher and writer agents."""

    def __init__(self, search_api_key: Optional[str] = None) -> None:
        super().__init__()
        search_tool = WebSearchTool(search_api_key) if search_api_key else None
        self.researcher = ResearchAgent(search_tool)
        self.writer = WriterAgent()

        # Coordination logic
        self.task_planner = dspy.Predict("request -> research_plan, writing_plan")

    async def close(self) -> None:
        """Clean up resources."""
        await self.researcher.close()

    async def forward(
        self, request: str, target_audience: str = "general", max_sources: int = 5
    ) -> Dict[str, Any]:
        """Execute full researcher -> writer workflow."""

        # Plan the task
        plan = self.task_planner(request=request)

        # Extract topic from request
        topic = request  # Could be enhanced with NLP to extract topic

        # Phase 1: Research
        research_results = await self.researcher.forward(
            topic=topic, max_sources=max_sources
        )

        # Phase 2: Writing
        writing_results = self.writer.forward(
            topic=topic,
            research_data=research_results["research_results"],
            key_points=research_results["key_points"],
            target_audience=target_audience,
        )

        # Combine results
        return {
            "request": request,
            "topic": topic,
            "research_phase": research_results,
            "writing_phase": writing_results,
            "final_article": writing_results["article"],
            "summary": writing_results["summary"],
            "sources": research_results["sources"],
        }
