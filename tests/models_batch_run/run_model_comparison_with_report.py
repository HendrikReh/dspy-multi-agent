#!/usr/bin/env python
"""Run model comparison test with comprehensive reporting."""
import asyncio
import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.integration.test_model_comparison import TestModelComparison
from src.utils.config import Config

async def main():
    # Load configuration from file
    config_path = Path(__file__).parent.parent.parent / "model_comparison_config.json"
    default_config = {
        "models": ["gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini"],
        "default_query": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
        "default_audience": "general audience",
        "default_sources": 5
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            print(f"Loaded configuration from {config_path}")
    else:
        config_data = default_config
        print(f"Configuration file not found at {config_path}, using defaults")
    
    parser = argparse.ArgumentParser(description="Run model comparison with comprehensive reporting")
    parser.add_argument(
        "--query",
        default=config_data.get("default_query", default_config["default_query"]),
        help="Test query to use"
    )
    parser.add_argument(
        "--audience",
        default=config_data.get("default_audience", default_config["default_audience"]),
        help="Target audience"
    )
    parser.add_argument(
        "--sources",
        type=int,
        default=config_data.get("default_sources", default_config["default_sources"]),
        help="Maximum number of sources"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=config_data.get("models", default_config["models"]),
        help="Models to compare"
    )
    parser.add_argument(
        "--config",
        help="Path to custom configuration file"
    )
    
    args = parser.parse_args()
    
    # If custom config file is specified, load it
    if args.config:
        custom_config_path = Path(args.config)
        if custom_config_path.exists():
            with open(custom_config_path, 'r') as f:
                custom_config = json.load(f)
                # Override with custom config values
                if "models" in custom_config and not any(arg.startswith("--models") for arg in sys.argv):
                    args.models = custom_config["models"]
                print(f"Loaded custom configuration from {custom_config_path}")
        else:
            print(f"Custom configuration file not found: {custom_config_path}")
    
    # Initialize test
    test_instance = TestModelComparison()
    config = Config()
    
    # Set up test queries
    test_queries = [{
        "request": args.query,
        "target_audience": args.audience,
        "max_sources": args.sources,
    }]
    
    # Override models if specified
    if args.models:
        original_method = test_instance.test_openai_models_comparison
        
        async def custom_test(config, test_queries, generate_report=True):
            # Temporarily override models
            test_instance.models = args.models
            test_instance.model_mapping = {m: m for m in args.models}
            return await original_method(config, test_queries, generate_report)
        
        test_instance.test_openai_models_comparison = custom_test
    
    print("="*80)
    print("Model Comparison Test with Comprehensive Reporting")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Query: {args.query}")
    print(f"Target Audience: {args.audience}")
    print(f"Max Sources: {args.sources}")
    print("="*80)
    print("\nStarting test...\n")
    
    # Run the test with report generation
    await test_instance.test_openai_models_comparison(config, test_queries, generate_report=True)
    
    print("\n" + "="*80)
    print("Test completed! Check the 'reports' directory for the comprehensive report.")
    print("="*80)

if __name__ == "__main__":
    # Install jinja2 if not available
    try:
        import jinja2
    except ImportError:
        print("Installing jinja2...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jinja2"])
    
    asyncio.run(main())