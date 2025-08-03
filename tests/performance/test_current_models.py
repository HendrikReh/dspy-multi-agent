#!/usr/bin/env python
"""Test with currently available models before o3/o4 release."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_model_series import ModelTestSeries
from utils.config import Config


async def main():
    """Test with current models."""
    config = Config()
    test_series = ModelTestSeries(config)
    
    # Test with currently available models
    current_models = ["gpt-4o-mini", "gpt-4o"]
    
    print("Testing with currently available models...")
    print("When o3/o4 are released, use test_model_series.py instead")
    print("-" * 60)
    
    # Override test queries to be simpler for current models
    test_series.test_queries = [
        {
            "id": "simple_test",
            "request": "Explain machine learning in simple terms",
            "target_audience": "general public",
            "max_sources": 3,
            "complexity": "low",
        },
        {
            "id": "medium_test",
            "request": "Compare supervised and unsupervised learning approaches",
            "target_audience": "students",
            "max_sources": 5,
            "complexity": "medium",
        },
    ]
    
    await test_series.run_test_series(current_models)


if __name__ == "__main__":
    asyncio.run(main())