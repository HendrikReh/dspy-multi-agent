# src/agents/writer.py
import dspy
from typing import List, Dict, Any
from signatures.agent_signatures import WritingSignature


class WriterAgent(dspy.Module):
    """Agent responsible for writing content based on research."""

    def __init__(self) -> None:
        super().__init__()
        self.writer = dspy.ChainOfThought(WritingSignature)
        self.editor = dspy.Predict("draft -> polished_article")

    def forward(
        self,
        topic: str,
        research_data: str,
        key_points: List[str],
        target_audience: str = "general",
        style: str = "informative",
    ) -> Dict[str, Any]:
        """Write content based on research findings."""

        # Generate initial draft
        draft_result = self.writer(
            topic=topic,
            research_data=research_data,
            key_points=key_points,
            target_audience=target_audience,
        )

        # Polish the article
        polished_article = self.editor(draft=draft_result.article)

        return {
            "topic": topic,
            "article": polished_article.polished_article,
            "summary": draft_result.summary,
            "target_audience": target_audience,
            "key_points_covered": key_points,
        }
