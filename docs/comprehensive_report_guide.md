# Comprehensive Report Generation Guide

## Overview

The model comparison framework generates comprehensive HTML reports that capture:
- Full LLM outputs from each model
- Performance metrics and visualizations
- Content similarity analysis
- All data in timestamped report directories

## Configuration

### Model Configuration File

Models are configured in `model_comparison_config.json`:

```json
{
  "models": ["gpt-4o-mini"],  // List of models to test
  "default_query": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
  "default_audience": "general audience",
  "default_sources": 5
}
```

### Running Tests with Reports

```bash
# Use default configuration (model_comparison_config.json)
uv run python run_model_comparison_with_report.py

# Use custom configuration file
uv run python run_model_comparison_with_report.py --config model_comparison_config.example.json

# Override configuration with command-line arguments
uv run python run_model_comparison_with_report.py --models gpt-4o o3-mini --query "Explain quantum computing"

# Full custom parameters
uv run python run_model_comparison_with_report.py \
  --models gpt-4o gpt-4o-mini o3 \
  --query "Explain artificial intelligence" \
  --audience "technical professionals" \
  --sources 10
```

## Report Structure

Each test run creates a timestamped directory under `reports/`:

```
reports/
└── model_comparison_YYYYMMDD_HHMMSS/
    ├── report_YYYYMMDD_HHMMSS.html     # Main HTML report
    ├── llm_outputs/                     # Full LLM responses
    │   ├── gpt-4o_1.txt
    │   ├── gpt-4o-mini_2.txt
    │   ├── o3_3.txt
    │   ├── o3-mini_4.txt
    │   └── o4-mini_5.txt
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

## Report Contents

### 1. HTML Report
The main report includes:
- **Executive Summary**: Model count, query count, fastest model
- **Performance Metrics**: Time to first token, total time, tokens/second
- **Visualizations**: Embedded charts and graphs
- **Model Comparisons**: Pairwise similarity and performance analysis
- **LLM Output References**: Links to full outputs

### 2. LLM Outputs
Each model's complete response is saved, containing:
- Model name and timestamp
- Original query
- **Full article content** (actual model output, not placeholders)

### 3. Visualizations

#### Performance Overview
- Bar charts comparing:
  - Time to first token
  - Tokens per second
  - Total generation time
  - Performance trade-off scatter plot

#### Speedup Matrix
- Heatmap showing relative performance between all model pairs
- Values > 1 indicate the row model is faster

#### Similarity Analysis
- Heatmap showing content similarity metrics:
  - Overall similarity
  - Key points overlap
  - Vocabulary overlap
  - Source overlap

#### Token Distribution
- Box plots showing token generation patterns by model

### 4. Data Files

#### complete_test_data.json
Contains all test data including:
- Models tested
- Queries used
- Full results with metrics
- Comparison data

#### metrics_summary.json
Condensed performance metrics:
- Model name
- Query preview
- All timing metrics

#### comparison_summary.json
Pairwise comparison results:
- Similarity scores
- Performance ratios

## Interpreting Results

### Performance Metrics
- **Time to First Token**: How quickly the model starts generating
  - GPT models: ~0.001s (nearly instant)
  - O3/o4 models: 5-30s (reasoning time)

- **Tokens per Second**: Generation speed
  - GPT models: 1000-2000+ tokens/sec
  - O3/o4 models: 40-100 tokens/sec

### Similarity Analysis
- **High similarity (>70%)**: Models produce very similar content
- **Medium similarity (30-70%)**: Some overlap but different approaches
- **Low similarity (<30%)**: Significantly different outputs

### Use Cases
- **Real-time applications**: Use GPT models for fast responses
- **Complex reasoning**: Use O3/o4 models for accuracy over speed
- **Cost optimization**: Balance performance vs API costs

## Custom Configuration Examples

### Testing Only Fast Models
Create `fast_models_config.json`:
```json
{
  "models": ["gpt-4o", "gpt-4o-mini"],
  "default_query": "Quick response test",
  "default_audience": "general",
  "default_sources": 3
}
```

Run: `uv run python run_model_comparison_with_report.py --config fast_models_config.json`

### Testing Reasoning Models
Create `reasoning_models_config.json`:
```json
{
  "models": ["o3", "o3-mini", "o4-mini"],
  "default_query": "Complex reasoning problem",
  "default_audience": "experts",
  "default_sources": 10
}
```

Run: `uv run python run_model_comparison_with_report.py --config reasoning_models_config.json`

## Tips

1. **Test Duration**: o3/o4 models take significantly longer (30-90s vs 1-3s for GPT models)
2. **Storage**: Each test creates ~1-5MB of data depending on output length
3. **Visualization**: Open the HTML report in a browser for interactive viewing
4. **LLM Outputs**: Check the `llm_outputs/` directory for complete model responses
5. **Configuration**: Use external config files to easily switch between different test scenarios