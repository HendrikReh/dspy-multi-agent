# src/agents/researcher.py
import dspy
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from signatures.agent_signatures import ResearchSignature
from tavily import AsyncTavilyClient
import os


class WebSearchTool:
    """Web search tool for research agents using Tavily API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", os.getenv("SEARCH_API_KEY"))
        if self.api_key:
            self.client = AsyncTavilyClient(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: No Tavily API key provided. Search functionality disabled.")

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information using Tavily API."""
        if not self.client:
            print("Search unavailable: No API key configured")
            return []
        
        try:
            # Perform search using Tavily
            response = await self.client.search(
                query=query,
                max_results=num_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                include_domains=[],
                exclude_domains=[]
            )
            
            # Format results to match expected structure
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("content", ""),
                    "url": result.get("url", ""),
                })
            
            # Add answer if available
            if response.get("answer"):
                results.insert(0, {
                    "title": "AI-Generated Summary",
                    "snippet": response["answer"],
                    "url": "tavily-answer"
                })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def close(self) -> None:
        """Close the Tavily client if needed."""
        # Tavily client handles its own cleanup
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
