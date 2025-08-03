#!/usr/bin/env python
"""Minimal test to generate actual results quickly."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config import Config
from agents.coordinator import MultiAgentCoordinator
import dspy


async def quick_test():
    """Run a quick test with one model and query."""
    config = Config()
    
    # Configure DSPy with gpt-4o-mini
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=config.openai_api_key,
        temperature=0.7,
        max_tokens=2000,
    )
    dspy.configure(lm=lm, async_max_workers=4)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(tavily_api_key=config.tavily_api_key)
    
    print("Running quick test with gpt-4o-mini...")
    
    # Simple query
    result = await coordinator.forward(
        request="Explain AI in one paragraph",
        target_audience="general public",
        max_sources=2
    )
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tests/integration/results/quick_test_{timestamp}.json"
    
    save_data = {
        "timestamp": timestamp,
        "model": "gpt-4o-mini",
        "query": "Explain AI in one paragraph",
        "result": {
            "final_article": result.get("final_article", ""),
            "summary": result.get("summary", ""),
            "sources": result.get("sources", []),
        }
    }
    
    Path("tests/integration/results").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Article preview: {result.get('final_article', '')[:200]}...")
    
    await coordinator.close()


if __name__ == "__main__":
    asyncio.run(quick_test())