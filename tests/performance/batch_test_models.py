#!/usr/bin/env python
"""Batch testing script for parallel model evaluation."""
import asyncio
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config import Config
from utils.model_configs import MODEL_CONFIGS, get_model_config
from utils.performance_metrics import PerformanceTracker
from agents.metrics_coordinator import MetricsCoordinator
import dspy


class BatchModelTester:
    """Run model tests in parallel batches."""
    
    def __init__(self, config: Config, max_parallel: int = 4):
        self.config = config
        self.max_parallel = max_parallel
        self.results = {}
        
    async def test_model_query(self, model_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single model-query combination."""
        model_config = get_model_config(model_name)
        
        # Create unique tracker for this test
        tracker = PerformanceTracker()
        
        try:
            # Configure DSPy
            lm = dspy.LM(
                model=f"openai/{model_config.api_name}",
                api_key=self.config.openai_api_key,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
            )
            dspy.configure(lm=lm, async_max_workers=1)  # Limit workers per model
            
            # Create coordinator
            coordinator = MetricsCoordinator(
                tavily_api_key=self.config.tavily_api_key,
                tracker=tracker
            )
            coordinator.set_model(model_name)
            
            # Run query
            result = await coordinator.forward(
                request=query["request"],
                target_audience=query.get("target_audience", "general"),
                max_sources=query.get("max_sources", 5),
            )
            
            # Clean up
            await coordinator.close()
            
            return {
                "status": "success",
                "model": model_name,
                "query_id": query.get("id", "unknown"),
                "result": result,
                "error": None,
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model": model_name,
                "query_id": query.get("id", "unknown"),
                "result": None,
                "error": str(e),
            }
    
    async def run_batch(self, batch: List[tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Run a batch of model-query tests in parallel."""
        tasks = []
        for model_name, query in batch:
            task = self.test_model_query(model_name, query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_name, query = batch[i]
                processed_results.append({
                    "status": "error",
                    "model": model_name,
                    "query_id": query.get("id", "unknown"),
                    "result": None,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def test_all_combinations(
        self, 
        models: List[str], 
        queries: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Test all model-query combinations with batching."""
        # Create all combinations
        combinations = [(model, query) for model in models for query in queries]
        total = len(combinations)
        
        print(f"Testing {len(models)} models × {len(queries)} queries = {total} combinations")
        print(f"Batch size: {self.max_parallel}")
        print("="*60)
        
        # Process in batches
        all_results = []
        start_time = time.time()
        
        for i in range(0, total, self.max_parallel):
            batch = combinations[i:i + self.max_parallel]
            batch_num = i // self.max_parallel + 1
            total_batches = (total + self.max_parallel - 1) // self.max_parallel
            
            print(f"\nBatch {batch_num}/{total_batches}")
            for model, query in batch:
                print(f"  - {model} × {query.get('id', 'unknown')}")
            
            # Run batch
            batch_start = time.time()
            batch_results = await self.run_batch(batch)
            batch_time = time.time() - batch_start
            
            all_results.extend(batch_results)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + len(batch), total)
            
            print(f"  Batch completed in {batch_time:.1f}s")
            
            # Show results summary
            success_count = sum(1 for r in batch_results if r["status"] == "success")
            print(f"  Success: {success_count}/{len(batch)}")
        
        total_time = time.time() - start_time
        
        # Organize results
        results_by_model = {}
        results_by_query = {}
        
        for result in all_results:
            model = result["model"]
            query_id = result["query_id"]
            
            if model not in results_by_model:
                results_by_model[model] = []
            results_by_model[model].append(result)
            
            if query_id not in results_by_query:
                results_by_query[query_id] = []
            results_by_query[query_id].append(result)
        
        return {
            "summary": {
                "total_tests": total,
                "successful": sum(1 for r in all_results if r["status"] == "success"),
                "failed": sum(1 for r in all_results if r["status"] == "error"),
                "total_time": total_time,
                "avg_time_per_test": total_time / total if total > 0 else 0,
            },
            "all_results": all_results,
            "by_model": results_by_model,
            "by_query": results_by_query,
        }


def create_test_matrix() -> tuple[List[str], List[Dict[str, Any]]]:
    """Create test matrix of models and queries."""
    # All o3/o4 variants
    models = [
        "o3", "o3-mini-low", "o3-mini-medium", "o3-mini-high",
        "o4", "o4-mini-low", "o4-mini-medium", "o4-mini-high"
    ]
    
    # Test queries of varying complexity
    queries = [
        {
            "id": "simple_explanation",
            "request": "Explain machine learning in one paragraph",
            "target_audience": "beginners",
            "max_sources": 2,
        },
        {
            "id": "technical_analysis",
            "request": "Analyze the computational complexity of transformer models",
            "target_audience": "ML researchers", 
            "max_sources": 5,
        },
        {
            "id": "creative_writing",
            "request": "Write a short story about AI discovering consciousness",
            "target_audience": "general readers",
            "max_sources": 0,
        },
        {
            "id": "research_synthesis",
            "request": "Synthesize recent advances in quantum computing applications",
            "target_audience": "technology professionals",
            "max_sources": 8,
        },
        {
            "id": "ethical_discussion",
            "request": "Discuss ethical considerations in autonomous vehicle decision-making",
            "target_audience": "policy makers",
            "max_sources": 6,
        },
    ]
    
    return models, queries


async def main():
    """Run batch testing."""
    parser = argparse.ArgumentParser(description="Batch test o3/o4 model variants")
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel tests (default: 4)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test"
    )
    parser.add_argument(
        "--output",
        default="batch_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = Config()
    
    # Create tester
    tester = BatchModelTester(config, max_parallel=args.parallel)
    
    # Get test matrix
    models, queries = create_test_matrix()
    
    if args.models:
        models = args.models
    
    print(f"Batch Testing Configuration:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Parallel tests: {args.parallel}")
    print(f"  Max CPU cores: {multiprocessing.cpu_count()}")
    
    # Model mapping - o3/o4 models are now available!
    model_mapping = {
        "o3": "o3",
        "o3-mini-low": "o3-mini",  # Using o3-mini as placeholder
        "o3-mini-medium": "o3-mini",  # Using o3-mini as placeholder
        "o3-mini-high": "o3-mini",  # Using o3-mini as placeholder
        "o4": "o4-mini",  # o4 doesn't exist, using o4-mini
        "o4-mini-low": "o4-mini",  # Using o4-mini as placeholder
        "o4-mini-medium": "o4-mini",  # Using o4-mini as placeholder
        "o4-mini-high": "o4-mini",  # Using o4-mini as placeholder
    }
    
    # Update model configs temporarily
    for model in models:
        if model in model_mapping:
            MODEL_CONFIGS[model].api_name = model_mapping[model]
    
    # Run tests
    results = await tester.test_all_combinations(models, queries)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH TEST SUMMARY")
    print("="*60)
    summary = results["summary"]
    print(f"Total tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_time']:.1f}s")
    print(f"Avg time per test: {summary['avg_time_per_test']:.1f}s")
    
    # Performance by model
    print("\nPerformance by Model:")
    for model in models:
        model_results = results["by_model"].get(model, [])
        success = sum(1 for r in model_results if r["status"] == "success")
        avg_time = sum(
            r["result"]["metrics"]["total_time"] 
            for r in model_results 
            if r["status"] == "success" and r["result"]
        ) / success if success > 0 else 0
        
        print(f"  {model}: {success}/{len(model_results)} successful, avg {avg_time:.1f}s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "parallel": args.parallel,
            "models": models,
            "queries": queries,
        },
        "results": results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())