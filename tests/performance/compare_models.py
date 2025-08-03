#!/usr/bin/env python
"""Script to compare o3 and o4 models (or other model pairs)."""
import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config import Config
# Import path will need adjustment when test_model_comparison.py exists
# from tests.integration.test_model_comparison import TestModelComparison
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.test_model_comparison import TestModelComparison


async def main():
    """Run model comparison."""
    parser = argparse.ArgumentParser(description="Compare LLM models with performance metrics")
    parser.add_argument(
        "--models", 
        nargs=2, 
        default=["gpt-4o-mini", "gpt-4o"],
        help="Two models to compare (default: gpt-4o-mini gpt-4o)"
    )
    parser.add_argument(
        "--query",
        default="Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
        help="Query to test with"
    )
    parser.add_argument(
        "--audience",
        default="general audience",
        help="Target audience for the article"
    )
    parser.add_argument(
        "--sources",
        type=int,
        default=5,
        help="Maximum number of sources to use"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = Config()
    
    # Create test instance
    test = TestModelComparison()
    
    # Create custom query
    custom_query = [{
        "request": args.query,
        "target_audience": args.audience,
        "max_sources": args.sources,
    }]
    
    # Temporarily override models in the test
    original_test = test.test_o3_vs_o4_comparison
    
    async def custom_test(config, test_queries):
        # Override models
        test.models = args.models
        await original_test(config, test_queries)
    
    test.test_o3_vs_o4_comparison = custom_test
    
    # Run comparison
    print(f"Comparing models: {args.models[0]} vs {args.models[1]}")
    print(f"Query: {args.query}")
    print(f"Target audience: {args.audience}")
    print(f"Max sources: {args.sources}")
    print("-" * 80)
    
    await test.test_o3_vs_o4_comparison(config, custom_query)


if __name__ == "__main__":
    asyncio.run(main())