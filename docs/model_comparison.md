# Model Comparison Guide

This guide explains how to use the model comparison framework to compare different LLM models with detailed performance metrics and comprehensive reporting.

## Overview

The model comparison system allows you to:
- Compare outputs from different LLM models on the same queries
- Track time-to-first-token and total generation time
- Measure tokens per second performance
- Analyze content similarity and differences
- Generate comprehensive HTML reports with visualizations
- Save full LLM outputs for detailed analysis

## Configuration

### Model Configuration File

The system uses `model_comparison_config.json` to define which models to test:

```json
{
  "models": ["gpt-4o-mini"],  // List of models to compare
  "default_query": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
  "default_audience": "general audience",
  "default_sources": 5
}
```

### Available Models

- **GPT Models**: `gpt-4o`, `gpt-4o-mini` (fast, general-purpose)
- **O3 Models**: `o3`, `o3-mini` (reasoning-focused, slower)
- **o4 Models**: `o4-mini` (advanced reasoning)

## Running Model Comparisons

### Basic Usage

```bash
# Run with default configuration
uv run python run_model_comparison_with_report.py

# Use custom configuration file
uv run python run_model_comparison_with_report.py --config custom_config.json

# Override specific parameters
uv run python run_model_comparison_with_report.py --models gpt-4o o3 --query "Explain quantum computing"
```

### Command-Line Options

- `--models`: List of models to compare (overrides config)
- `--query`: Test query to use (overrides config)
- `--audience`: Target audience (default: "general audience")
- `--sources`: Maximum number of sources (default: 5)
- `--config`: Path to custom configuration file

### Examples

```bash
# Compare only GPT models
uv run python run_model_comparison_with_report.py --models gpt-4o gpt-4o-mini

# Test with custom query and audience
uv run python run_model_comparison_with_report.py \
  --query "Explain machine learning algorithms" \
  --audience "computer science students" \
  --sources 10

# Use a specific configuration
uv run python run_model_comparison_with_report.py --config configs/reasoning_models.json
```

## Generated Reports

### Report Structure

Each test run creates a timestamped directory:

```
reports/
└── model_comparison_YYYYMMDD_HHMMSS/
    ├── report_YYYYMMDD_HHMMSS.html     # Main HTML report
    ├── llm_outputs/                     # Full LLM responses
    │   └── model_name_N.txt             # Complete output for each model
    ├── visualizations/                  # Generated charts
    │   ├── performance_overview.png
    │   ├── speedup_matrix.png
    │   ├── similarity_heatmap.png
    │   └── token_distribution.png
    └── data/                           # Raw data files
        ├── complete_test_data.json
        ├── metrics_summary.json
        └── comparison_summary.json
```

### Report Contents

1. **Executive Summary**: Quick overview of test results
2. **Performance Metrics**: Detailed timing and token analysis
3. **Visualizations**: Interactive charts and graphs
4. **Model Comparisons**: Pairwise similarity analysis
5. **Full LLM Outputs**: Complete responses from each model

## Metrics Explained

### Performance Metrics
- **Time to First Token**: Response latency
  - GPT models: ~0.001s
  - O3/o4 models: 5-30s (includes reasoning time)
- **Total Time**: Complete generation time
- **Tokens per Second**: Generation speed
  - GPT models: 1000-2000+ tokens/sec
  - O3/o4 models: 40-100 tokens/sec

### Content Comparison Metrics
- **Overall Similarity**: Text similarity score (0-1)
- **Key Points Overlap**: Shared main ideas
- **Vocabulary Overlap**: Common significant words
- **Source Overlap**: Shared references

### Quality Metrics
- **Sentence Count**: Output length
- **Average Sentence Length**: Readability indicator
- **Complexity Ratio**: Percentage of complex words

## Programmatic Usage

```python
from tests.integration.test_model_comparison import TestModelComparison
from utils.config import Config

# Initialize
test_instance = TestModelComparison()
config = Config()

# Define test queries
test_queries = [{
    "request": "Your question here",
    "target_audience": "general audience",
    "max_sources": 5
}]

# Run comparison with report generation
await test_instance.test_openai_models_comparison(
    config, 
    test_queries, 
    generate_report=True
)
```

## Custom Configuration Examples

### Fast Models Only
```json
{
  "models": ["gpt-4o", "gpt-4o-mini"],
  "default_query": "Quick test query",
  "default_audience": "general",
  "default_sources": 3
}
```

### Reasoning Models Only
```json
{
  "models": ["o3", "o3-mini", "o4-mini"],
  "default_query": "Complex reasoning problem",
  "default_audience": "experts",
  "default_sources": 10
}
```

### Mixed Performance Test
```json
{
  "models": ["gpt-4o-mini", "o3-mini"],
  "default_query": "Balanced test query",
  "default_audience": "professionals",
  "default_sources": 5
}
```

## Tips and Best Practices

1. **Model Selection**:
   - Use GPT models for real-time applications
   - Use O3/o4 models for complex reasoning tasks
   - Test both for comprehensive comparison

2. **Performance Considerations**:
   - o3/o4 models take 10-50x longer than GPT models
   - Plan test duration accordingly
   - Consider running overnight for large test suites

3. **Report Analysis**:
   - Open HTML reports in a browser for best viewing
   - Check similarity scores for output consistency
   - Review full LLM outputs for quality assessment

4. **Storage Management**:
   - Each test creates 1-5MB of data
   - Clean up old reports periodically
   - Archive important comparisons

## Troubleshooting

- **API Errors**: Ensure all model names are valid and API keys are set
- **Timeout Issues**: o3/o4 models may need longer timeouts
- **Memory Usage**: Large comparisons may require more RAM
- **Missing Visualizations**: Install matplotlib and seaborn if needed