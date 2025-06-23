# src/agents/researcher.py
import dspy
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from signatures.agent_signatures import ResearchSignature


class WebSearchTool:
    """Web search tool for research agents."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self.session = httpx.AsyncClient()

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information."""
        # Example using a hypothetical search API
        # Replace with actual search service (You.com, Serper, etc.)
        try:
            response = await self.session.get(
                "https://api.search.example.com/search",
                params={"q": query, "count": num_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            results = response.json().get("results", [])
            return [
                {
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "url": result.get("url", ""),
                }
                for result in results
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def close(self) -> None:
        """Explicitly close the HTTP session."""
        if hasattr(self, "session") and not self.session.is_closed:
            await self.session.aclose()

    def __del__(self) -> None:
        """Clean up the HTTP session."""
        # Don't try to close async resources in __del__ as it's unreliable
        # The session will be cleaned up by garbage collection
        pass


class ResearchAgent(dspy.Module):
    """Agent responsible for conducting research on given topics."""

    def __init__(self, search_tool: Optional[WebSearchTool] = None) -> None:
        super().__init__()
        self.search_tool = search_tool or WebSearchTool()
        self.researcher = dspy.ChainOfThought(ResearchSignature)
        self._owns_search_tool = search_tool is None  # Track if we created the tool

    async def close(self) -> None:
        """Close the search tool if we own it."""
        if self._owns_search_tool and self.search_tool:
            await self.search_tool.close()

    async def forward(
        self, topic: str, context: str = "", max_sources: int = 5
    ) -> Dict[str, Any]:
        """Conduct research on the given topic."""
        # Gather web search results
        search_results = await self.search_tool.search(topic, max_sources)

        # Create context from search results
        search_context = "\n".join(
            [
                f"Source: {result['title']}\n{result['snippet']}\nURL: {result['url']}\n"
                for result in search_results
            ]
        )

        full_context = f"{context}\n\nWeb Search Results:\n{search_context}"

        # Conduct research using DSPy
        research_result = self.researcher(topic=topic, context=full_context)

        return {
            "topic": topic,
            "research_results": research_result.research_results,
            "key_points": research_result.key_points,
            "sources": research_result.sources,
            "search_results": search_results,
            "context": full_context,
        }
