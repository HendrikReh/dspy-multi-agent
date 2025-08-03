# Visualization Guide

## Overview

The model comparison framework automatically generates visualizations as part of the comprehensive HTML reports. These visualizations help analyze performance differences and content similarity between models.

## Automatic Report Generation

When running model comparisons with report generation enabled, visualizations are automatically created:

```bash
# Run with automatic visualization generation
uv run python run_model_comparison_with_report.py
```

This creates a timestamped report directory with all visualizations:

```
reports/model_comparison_YYYYMMDD_HHMMSS/
├── report_YYYYMMDD_HHMMSS.html     # Contains embedded visualizations
├── visualizations/
│   ├── performance_overview.png     # Comprehensive performance metrics
│   ├── speedup_matrix.png          # Model-to-model speedup comparison
│   ├── similarity_heatmap.png      # Content similarity analysis
│   └── token_distribution.png      # Token generation patterns
└── data/                           # Raw data for custom analysis
```

## Generated Visualizations

### 1. Performance Overview
**File**: `performance_overview.png`

Multi-panel visualization showing:
- Time to first token comparison (bar chart)
- Tokens per second comparison (bar chart)
- Total generation time comparison (bar chart)
- Performance trade-off scatter plot (time vs tokens/sec)

### 2. Speedup Matrix
**File**: `speedup_matrix.png`

Heatmap showing relative performance between all model pairs:
- Values > 1.0: Row model is faster than column model
- Values < 1.0: Column model is faster than row model
- Color intensity indicates magnitude of difference

### 3. Similarity Heatmap
**File**: `similarity_heatmap.png`

Content similarity analysis between models:
- Overall similarity score (0-1)
- Key points overlap percentage
- Vocabulary overlap percentage
- Source overlap percentage

### 4. Token Distribution
**File**: `token_distribution.png`

Box plots showing token generation patterns:
- Distribution of tokens generated across queries
- Median, quartiles, and outliers for each model
- Helps identify consistency in output length

## Interpreting Visualizations

### Performance Patterns

**GPT Models (gpt-4o, gpt-4o-mini)**:
- Near-instant time to first token (~0.001s)
- High tokens per second (1000-2000+)
- Consistent performance across queries

**O3/o4 Models**:
- Significant reasoning time (5-30s to first token)
- Lower tokens per second (40-100)
- More variable performance based on query complexity

### Similarity Analysis

**High Similarity (>70%)**:
- Models produce very similar content
- Good for consistency requirements
- May indicate redundant model selection

**Medium Similarity (30-70%)**:
- Different approaches to same problem
- Good for diverse perspectives
- Optimal for ensemble approaches

**Low Similarity (<30%)**:
- Significantly different outputs
- Check for quality differences
- May indicate model specialization

## Custom Visualization

### Using Raw Data

The `data/` directory contains JSON files for custom analysis:

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open('reports/model_comparison_*/data/complete_test_data.json') as f:
    data = json.load(f)

# Custom visualization code
# Access data['results'], data['comparisons'], etc.
```

### Legacy Visualization Tools

For backward compatibility, standalone visualization scripts are available:

```bash
# For model comparison results
uv run python tests/tools/visualize_comparison_results.py <json_file>

# For test series results
uv run python tests/tools/visualize_results.py <json_file>
```

## Tips for Analysis

1. **Compare Like Models**: Group GPT models separately from O3/o4 models for fair comparison
2. **Consider Use Case**: Fast models for real-time, reasoning models for complex tasks
3. **Check Outliers**: Token distribution outliers may indicate edge cases
4. **Validate Similarity**: Low similarity doesn't mean poor quality - check actual outputs
5. **Performance vs Quality**: Always balance speed metrics with output quality

## Exporting Visualizations

All visualizations are:
- Saved as high-resolution PNG files
- Embedded in the HTML report as base64
- Available for separate use in presentations
- Generated with consistent styling for reports