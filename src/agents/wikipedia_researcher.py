# src/agents/wikipedia_researcher.py
import dspy
from typing import List, Dict, Any


def search_wikipedia(query: str, k: int = 5) -> List[str]:
    """Wikipedia search tool for DSPy agents."""
    retriever = dspy.Retrieve(k=k)
    results = retriever(query).passages
    return list(results)  # Ensure we return a list


def lookup_wikipedia(title: str) -> str:
    """Lookup specific Wikipedia article."""
    # Implementation would use Wikipedia API
    # This is a placeholder
    return f"Wikipedia content for: {title}"


class WikipediaResearchAgent(dspy.Module):
    """Research agent using Wikipedia as primary source."""

    def __init__(self) -> None:
        super().__init__()
        self.tools = [search_wikipedia, lookup_wikipedia]
        self.agent = dspy.ReAct(
            "topic -> research_findings", tools=self.tools, max_iters=5
        )

    def forward(self, topic: str) -> Dict[str, Any]:
        """Research using Wikipedia sources."""
        result = self.agent(topic=topic)

        return {
            "topic": topic,
            "findings": result.research_findings,
            "trajectory": result.trajectory,
            "sources": ["Wikipedia"] * len(result.trajectory),
        }
