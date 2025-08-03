#!/usr/bin/env python
"""Comprehensive test series for o3/o4 model variants."""
import asyncio
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from tabulate import tabulate
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config import Config
from utils.model_configs import MODEL_CONFIGS, get_model_config, get_test_pairs, get_capability_settings
from utils.performance_metrics import PerformanceTracker
from utils.output_comparison import compare_outputs, calculate_similarity
from agents.metrics_coordinator import MetricsCoordinator
import dspy


class ModelTestSeries:
    """Run comprehensive tests across all model variants."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tracker = PerformanceTracker()
        self.results = []
        self.test_queries = [
            {
                "id": "tech_simple",
                "request": "Explain how blockchain technology works in simple terms",
                "target_audience": "general public",
                "max_sources": 3,
                "complexity": "low",
            },
            {
                "id": "tech_complex", 
                "request": "Analyze the technical challenges and solutions for implementing post-quantum cryptography in existing blockchain systems",
                "target_audience": "cryptography researchers",
                "max_sources": 10,
                "complexity": "high",
            },
            {
                "id": "science_medium",
                "request": "Discuss recent breakthroughs in CRISPR gene editing and their ethical implications",
                "target_audience": "science journalists",
                "max_sources": 7,
                "complexity": "medium",
            },
            {
                "id": "business_analysis",
                "request": "Evaluate the impact of AI on workforce transformation in the financial services industry",
                "target_audience": "business executives",
                "max_sources": 8,
                "complexity": "medium",
            },
            {
                "id": "creative_task",
                "request": "Write a compelling narrative about the future of human-AI collaboration in creative industries",
                "target_audience": "creative professionals",
                "max_sources": 5,
                "complexity": "medium",
            },
        ]
    
    async def test_model(self, model_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Test a single model with a query."""
        model_config = get_model_config(model_name)
        
        print(f"\n  Testing {model_name} (capability: {model_config.capability.value})...")
        
        try:
            # Get capability-specific settings
            capability_settings = get_capability_settings(model_config.capability)
            
            # Configure DSPy with model-specific settings
            # Note: When o3/o4 are available, they may support reasoning_effort parameter
            lm_kwargs = {
                "model": f"openai/{model_config.api_name}",
                "api_key": self.config.openai_api_key,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
            }
            
            # Add capability-specific parameters if supported
            if model_config.family.value in ["o3", "o4"]:
                lm_kwargs.update(capability_settings)
            
            lm = dspy.LM(**lm_kwargs)
            dspy.configure(lm=lm, async_max_workers=self.config.async_workers)
            
            # Create coordinator
            coordinator = MetricsCoordinator(
                tavily_api_key=self.config.tavily_api_key,
                tracker=self.tracker
            )
            coordinator.set_model(model_name)
            
            # Run query
            start_time = time.time()
            result = await coordinator.forward(
                request=query["request"],
                target_audience=query["target_audience"],
                max_sources=query["max_sources"],
            )
            
            # Add query info to result
            result["query_id"] = query["id"]
            result["model_config"] = {
                "name": model_config.name,
                "family": model_config.family.value,
                "size": model_config.size.value,
                "capability": model_config.capability.value,
            }
            
            # Clean up
            await coordinator.close()
            
            return result
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            return None
    
    async def run_test_series(self, models: Optional[List[str]] = None):
        """Run tests across all models."""
        if models is None:
            # Test all o3/o4 variants
            models = [
                "o3", "o3-mini-low", "o3-mini-medium", "o3-mini-high",
                "o4", "o4-mini-low", "o4-mini-medium", "o4-mini-high"
            ]
        
        # Model mapping - o3/o4 models are now available!
        # Note: o3-mini-low/medium/high and o4-mini-low/medium/high variants don't exist yet
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
        
        print(f"Running test series with {len(models)} models and {len(self.test_queries)} queries")
        print("="*80)
        
        # Run tests
        for query in self.test_queries:
            print(f"\nQuery: {query['id']} - Complexity: {query['complexity']}")
            print(f"Request: {query['request'][:100]}...")
            
            query_results = {}
            
            for model in models:
                # Use mapped model until o3/o4 are available
                actual_model = model_mapping.get(model, model)
                if actual_model != model:
                    print(f"  Note: Using {actual_model} as placeholder for {model}")
                
                result = await self.test_model(model, query)
                if result:
                    query_results[model] = result
            
            # Store results
            self.results.append({
                "query": query,
                "model_results": query_results,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("TEST SERIES REPORT")
        print("="*80)
        
        # Performance summary table
        perf_data = []
        
        for result in self.results:
            query_id = result["query"]["id"]
            
            for model_name, model_result in result["model_results"].items():
                if model_result and "metrics" in model_result:
                    metrics = model_result["metrics"]
                    config = model_result["model_config"]
                    
                    perf_data.append({
                        "Model": model_name,
                        "Family": config["family"],
                        "Capability": config["capability"],
                        "Query": query_id,
                        "First Token (s)": f"{metrics['time_to_first_token']:.3f}",
                        "Total Time (s)": f"{metrics['total_time']:.2f}",
                        "Tokens/sec": f"{metrics['tokens_per_second']:.1f}",
                    })
        
        if perf_data:
            df = pd.DataFrame(perf_data)
            
            # Group by model and calculate averages
            print("\n### Average Performance by Model")
            avg_perf = df.groupby(["Model", "Family", "Capability"]).agg({
                "First Token (s)": lambda x: f"{pd.to_numeric(x).mean():.3f}",
                "Total Time (s)": lambda x: f"{pd.to_numeric(x).mean():.2f}",
                "Tokens/sec": lambda x: f"{pd.to_numeric(x).mean():.1f}",
            }).reset_index()
            
            print(tabulate(avg_perf, headers="keys", tablefmt="grid"))
            
            # Performance by query complexity
            print("\n### Performance by Query Complexity")
            for query in self.test_queries:
                query_df = df[df["Query"] == query["id"]]
                if not query_df.empty:
                    print(f"\n{query['id']} (Complexity: {query['complexity']})")
                    print(tabulate(query_df[["Model", "First Token (s)", "Total Time (s)", "Tokens/sec"]], 
                                 headers="keys", tablefmt="grid"))
        
        # Model comparison pairs
        print("\n### Model Pair Comparisons")
        self.compare_model_pairs()
        
        # Save detailed results
        self.save_results()
    
    def compare_model_pairs(self):
        """Compare interesting model pairs."""
        pairs = [
            ("o3", "o4"),
            ("o3-mini-low", "o3-mini-high"),
            ("o4-mini-low", "o4-mini-high"),
            ("o3-mini-medium", "o4-mini-medium"),
        ]
        
        for model_a, model_b in pairs:
            print(f"\n{model_a} vs {model_b}")
            
            similarities = []
            perf_ratios = []
            
            for result in self.results:
                if model_a in result["model_results"] and model_b in result["model_results"]:
                    result_a = result["model_results"][model_a]
                    result_b = result["model_results"][model_b]
                    
                    # Content similarity
                    if result_a and result_b:
                        similarity = calculate_similarity(
                            result_a.get("final_article", ""),
                            result_b.get("final_article", "")
                        )
                        similarities.append(similarity)
                        
                        # Performance ratio
                        if "metrics" in result_a and "metrics" in result_b:
                            time_a = result_a["metrics"]["total_time"]
                            time_b = result_b["metrics"]["total_time"]
                            perf_ratios.append(time_a / time_b if time_b > 0 else 0)
            
            if similarities:
                print(f"  Avg content similarity: {sum(similarities)/len(similarities):.2%}")
            if perf_ratios:
                avg_ratio = sum(perf_ratios)/len(perf_ratios)
                faster_model = model_b if avg_ratio > 1 else model_a
                print(f"  Performance: {faster_model} is {abs(1-avg_ratio):.1%} faster on average")
    
    def save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(__file__).parent.parent / "integration" / "results" / f"model_series_{timestamp}.json"
        
        # Create directory
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        save_data = {
            "timestamp": timestamp,
            "models_tested": list(set(
                model for r in self.results 
                for model in r["model_results"].keys()
            )),
            "queries": self.test_queries,
            "results": self.results,
            "performance_summary": self.get_performance_summary(),
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {filename}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        model_stats = {}
        
        for result in self.results:
            for model_name, model_result in result["model_results"].items():
                if model_result and "metrics" in model_result:
                    metrics = model_result["metrics"]
                    
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            "first_token_times": [],
                            "total_times": [],
                            "tokens_per_sec": [],
                        }
                    
                    model_stats[model_name]["first_token_times"].append(metrics["time_to_first_token"])
                    model_stats[model_name]["total_times"].append(metrics["total_time"])
                    model_stats[model_name]["tokens_per_sec"].append(metrics["tokens_per_second"])
        
        # Calculate statistics
        summary = {}
        for model, stats in model_stats.items():
            summary[model] = {
                "avg_first_token": sum(stats["first_token_times"]) / len(stats["first_token_times"]),
                "avg_total_time": sum(stats["total_times"]) / len(stats["total_times"]),
                "avg_tokens_per_sec": sum(stats["tokens_per_sec"]) / len(stats["tokens_per_sec"]),
                "min_first_token": min(stats["first_token_times"]),
                "max_first_token": max(stats["first_token_times"]),
            }
        
        return summary


async def main():
    """Run the test series."""
    parser = argparse.ArgumentParser(description="Test series for o3/o4 model variants")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: all o3/o4 variants)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer models"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = Config()
    
    # Create test series
    test_series = ModelTestSeries(config)
    
    # Determine models to test
    if args.quick:
        models = ["o3", "o4", "o3-mini-medium", "o4-mini-medium"]
    else:
        models = args.models
    
    # Run tests
    await test_series.run_test_series(models)


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import pandas
        import tabulate
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "tabulate"])
    
    asyncio.run(main())