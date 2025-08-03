# tests/integration/test_model_comparison.py
"""End-to-end test comparing OpenAI models with performance metrics."""
import pytest
import asyncio
import os
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import dspy
from agents.metrics_coordinator import MetricsCoordinator
from utils.performance_metrics import PerformanceTracker, ModelComparison
from utils.output_comparison import compare_outputs, print_comparison_summary, generate_diff
from utils.config import Config
from utils.report_generator import ComprehensiveReportGenerator


@pytest.mark.integration
@pytest.mark.asyncio
class TestModelComparison:
    """Test comparing OpenAI models end-to-end."""
    
    @pytest.fixture
    def config(self):
        """Get configuration."""
        return Config()
    
    @pytest.fixture
    def test_queries(self):
        """Test queries for comparison."""
        return [
            {
                "request": "Write a comprehensive article about the latest advances in quantum computing and their potential impact on cryptography",
                "target_audience": "technology professionals",
                "max_sources": 5,
            },
            {
                "request": "Explain the role of artificial intelligence in climate change mitigation strategies",
                "target_audience": "policy makers",
                "max_sources": 7,
            },
            {
                "request": "Analyze the current state and future prospects of renewable energy technologies",
                "target_audience": "investors and business leaders",
                "max_sources": 10,
            },
        ]
    
    async def run_model_test(self, model_name: str, query: dict, config: Config, tracker: PerformanceTracker, display_name: str = None, report_generator=None) -> dict:
        """Run a single model test."""
        display_name = display_name or model_name
        print(f"\n{'='*60}")
        print(f"Testing {display_name} with query: {query['request'][:60]}...")
        print(f"{'='*60}")
        
        # Configure DSPy with specific model
        # Special handling for o3/o4 models which require temperature=1.0 and max_tokens>=20000
        if model_name.startswith(('o3', 'o4')):
            lm = dspy.LM(
                model=f"openai/{model_name}",
                api_key=config.openai_api_key,
                temperature=1.0,
                max_tokens=25000,
            )
        else:
            lm = dspy.LM(
                model=f"openai/{model_name}",
                api_key=config.openai_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        dspy.configure(lm=lm, async_max_workers=config.async_workers)
        
        # Create coordinator with metrics tracking
        coordinator = MetricsCoordinator(
            tavily_api_key=config.tavily_api_key,
            tracker=tracker
        )
        coordinator.set_model(display_name)
        
        try:
            # Run the query
            result = await coordinator.forward(
                request=query["request"],
                target_audience=query["target_audience"],
                max_sources=query["max_sources"],
            )
            
            # Capture full LLM output if report generator is active
            if report_generator and result:
                # Use the actual final article as the LLM output
                llm_output = result.get('final_article', 'No article generated')
                report_generator.add_test_result(display_name, query, result, llm_output)
            
            # Print basic metrics
            metrics = result["metrics"]
            print(f"\nPerformance Metrics for {display_name}:")
            print(f"  Total time: {metrics['total_time']:.2f}s")
            print(f"  Time to first token: {metrics['time_to_first_token']:.2f}s")
            print(f"  Tokens generated: {metrics['tokens_generated']}")
            print(f"  Tokens per second: {metrics['tokens_per_second']:.2f}")
            print(f"  Phase timings:")
            for phase, timing in metrics['phase_timings'].items():
                print(f"    {phase}: {timing:.2f}s")
            
            # Clean up
            await coordinator.close()
            
            return result
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            await coordinator.close()
            raise
    
    async def test_openai_models_comparison(self, config, test_queries, generate_report=True):
        """Compare OpenAI models on multiple queries."""
        # Skip if no API key
        if not config.openai_api_key:
            pytest.skip("OpenAI API key not configured")
        
        # Suppress dspy forward() warnings
        warnings.filterwarnings("ignore", message=r".*forward\(\) directly is discouraged.*")
        
        # Initialize report generator if requested
        report_generator = ComprehensiveReportGenerator() if generate_report else None
        
        # Models to compare: gpt-4o, gpt-4o-mini, o4-mini, o3-mini, o3
        models = ["gpt-4o", "gpt-4o-mini", "o4-mini", "o3-mini", "o3"]
        
        # Model mapping - all models are now available!
        model_mapping = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "o4-mini": "o4-mini",
            "o3-mini": "o3-mini",
            "o3": "o3",
        }
        
        tracker = PerformanceTracker()
        all_comparisons = []
        
        for query_idx, query in enumerate(test_queries):
            print(f"\n{'#'*80}")
            print(f"QUERY {query_idx + 1}/{len(test_queries)}")
            print(f"{'#'*80}")
            
            results = {}
            
            # Run each model
            for model in models:
                try:
                    # Use mapped model if original is not available
                    actual_model = model_mapping.get(model, model)
                    if actual_model != model:
                        print(f"\nNote: Using {actual_model} as placeholder for {model}")
                    
                    # Run test with actual model but store under original name
                    result = await self.run_model_test(actual_model, query, config, tracker, display_name=model, report_generator=report_generator)
                    results[model] = result
                    
                except Exception as e:
                    print(f"Failed to test {model}: {str(e)}")
                    continue
            
            # Compare outputs between all model pairs
            if len(results) >= 2:
                # Compare each model with others
                model_list = list(results.keys())
                for i in range(len(model_list)):
                    for j in range(i + 1, len(model_list)):
                        model_a, model_b = model_list[i], model_list[j]
                        comparison = compare_outputs(
                            results[model_a],
                            results[model_b],
                            model_a,
                            model_b
                        )
                        
                        # Print comparison summary
                        print_comparison_summary(comparison)
                        
                        # Generate diff for detailed comparison
                        diff = generate_diff(
                            results[model_a]["final_article"],
                            results[model_b]["final_article"],
                            model_a,
                            model_b
                        )
                        
                        # Store comparison
                        all_comparisons.append({
                            "query": query,
                            "models": [model_a, model_b],
                            "comparison": comparison,
                            "diff": diff,
                        })
                        
                        # Add to report generator
                        if report_generator:
                            report_generator.add_comparison(model_a, model_b, comparison)
        
        # Overall performance comparison
        if len(all_comparisons) > 0:
            print(f"\n{'='*80}")
            print("OVERALL PERFORMANCE COMPARISON")
            print(f"{'='*80}")
            
            for model in models:
                avg_metrics = tracker.get_average_metrics(model)
                if avg_metrics:
                    print(f"\n{model} Average Metrics:")
                    print(f"  Avg time to first token: {avg_metrics.time_to_first_token:.2f}s")
                    print(f"  Avg total time: {avg_metrics.total_time:.2f}s")
                    print(f"  Avg tokens/second: {avg_metrics.tokens_per_second:.2f}")
            
            # Model comparisons for all pairs
            print(f"\nMODEL PAIR COMPARISONS:")
            model_comparisons = []
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    try:
                        model_a, model_b = models[i], models[j]
                        model_comparison = tracker.compare_models(model_a, model_b)
                        comparison_dict = model_comparison.to_dict()
                        
                        print(f"\n{model_a} vs {model_b}:")
                        print(f"  First token speedup: {comparison_dict['comparison']['speedup_first_token']:.2f}x")
                        print(f"  Total time speedup: {comparison_dict['comparison']['speedup_total']:.2f}x")
                        
                        model_comparisons.append(comparison_dict)
                        
                    except Exception as e:
                        print(f"Could not compare {model_a} and {model_b}: {str(e)}")
            
            # Save detailed results
            self.save_results(all_comparisons, model_comparisons)
            
            # Generate comprehensive report
            if report_generator:
                report_path = report_generator.generate_html_report()
                print(f"\nComprehensive report available at: {report_path}")
    
    def save_results(self, comparisons: list, performance_comparisons: list):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tests/integration/results/model_comparison_{timestamp}.json"
        
        # Create results directory
        os.makedirs("tests/integration/results", exist_ok=True)
        
        results = {
            "timestamp": timestamp,
            "models_tested": ["gpt-4o", "gpt-4o-mini", "o4-mini", "o3-mini", "o3"],
            "query_comparisons": comparisons,
            "performance_comparisons": performance_comparisons,
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
    
    async def test_streaming_metrics(self, config):
        """Test streaming response metrics tracking."""
        if not config.openai_api_key:
            pytest.skip("OpenAI API key not configured")
        
        # This test demonstrates how streaming metrics would work
        # In a real implementation, DSPy would need to support streaming
        print("\nNote: Streaming metrics test requires DSPy streaming support")
        print("This is a placeholder for future implementation")
        
        # Example of how it would work:
        # tracker = PerformanceTracker()
        # coordinator = MetricsCoordinator(config.tavily_api_key, tracker)
        # 
        # # Get streaming response
        # stream = await coordinator.forward_stream(
        #     request="Test query",
        #     stream=True
        # )
        # 
        # # Wrap with metrics tracking
        # metrics_stream = StreamingMetricsWrapper(
        #     stream, tracker, "test_op", "gpt-4"
        # )
        # 
        # # Process stream
        # async for chunk in metrics_stream:
        #     print(chunk.content, end='', flush=True)


@pytest.mark.integration
def test_comparison_utilities():
    """Test the comparison utilities work correctly."""
    from utils.output_comparison import calculate_similarity, extract_key_points, compare_key_points
    
    # Test similarity calculation
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quick brown fox leaps over the lazy dog."
    similarity = calculate_similarity(text1, text2)
    assert 0.8 < similarity < 1.0, f"Expected high similarity, got {similarity}"
    
    # Test key point extraction
    text_with_points = """
    Here are the main points:
    1. First important point
    2. Second important point
    - Third point in bullet form
    * Fourth point with asterisk
    """
    points = extract_key_points(text_with_points)
    assert len(points) == 4, f"Expected 4 points, got {len(points)}"
    
    # Test key point comparison
    points1 = ["AI is transforming healthcare", "Machine learning improves diagnosis"]
    points2 = ["AI is transforming healthcare", "Deep learning revolutionizes treatment"]
    comparison = compare_key_points(points1, points2)
    assert len(comparison["common_points"]) == 1
    assert len(comparison["unique_to_first"]) == 1
    assert len(comparison["unique_to_second"]) == 1
    
    print("✓ Comparison utilities tests passed")


if __name__ == "__main__":
    # Run the comparison test
    test_instance = TestModelComparison()
    config = Config()
    
    # Default test queries
    test_queries = [
        {
            "request": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
            "target_audience": "general audience",
            "max_sources": 5,
        }
    ]
    
    # Run with report generation enabled
    asyncio.run(test_instance.test_openai_models_comparison(config, test_queries, generate_report=True))