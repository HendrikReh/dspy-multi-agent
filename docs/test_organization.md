# Test Organization Guide

This document describes the organization of test files in the DSPy multi-agent project.

## Directory Structure

All test files have been organized into the following structure:

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and shared fixtures
├── run_tests.py                   # Main test runner script
├── unit/                          # Unit tests for individual components
│   ├── __init__.py
│   ├── test_config.py             # Config class tests
│   ├── test_coordinator.py        # MultiAgentCoordinator tests
│   ├── test_research_agent.py     # ResearchAgent tests
│   ├── test_web_search_tool.py    # WebSearchTool tests
│   └── test_writer_agent.py       # WriterAgent tests
├── integration/                   # Integration tests
│   ├── __init__.py
│   ├── test_model_comparison.py   # Model comparison tests
│   └── results/                   # Test result JSON files
├── performance/                   # Performance and model testing
│   ├── test_model_series.py       # Comprehensive o3/o4 model tests
│   ├── batch_test_models.py       # Parallel batch testing
│   ├── compare_models.py          # Model comparison script
│   ├── test_current_models.py     # Test with available models
│   ├── minimal_test.py            # Quick minimal test
│   └── quick_test_demo.py         # Demo showing result locations
└── tools/                         # Testing tools and utilities
    └── visualize_results.py       # Visualization tool for test results
```

## Running Tests

### Unit Tests
```bash
# Run all unit tests
uv run pytest tests/unit/

# Run with coverage
uv run pytest tests/unit/ --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_coordinator.py
```

### Performance Tests
```bash
# Run model comparison with report generation (recommended)
uv run python run_model_comparison_with_report.py

# Use custom configuration
uv run python run_model_comparison_with_report.py --config model_comparison_config.json

# Run integration tests directly
uv run pytest tests/integration/test_model_comparison.py -v

# Run minimal test
uv run python tests/performance/minimal_test.py

# Run model comparison script
uv run python tests/performance/compare_models.py --models gpt-4o-mini gpt-4o
```

### Visualization
```bash
# Visualize test results
uv run python tests/tools/visualize_results.py tests/integration/results/model_series_20250803_134102.json
```

## Test Results

Test results are saved in multiple formats:

### Report Directory Structure
```
reports/
└── model_comparison_YYYYMMDD_HHMMSS/
    ├── report_YYYYMMDD_HHMMSS.html     # Comprehensive HTML report
    ├── llm_outputs/                     # Full LLM responses
    ├── visualizations/                  # Performance charts
    └── data/                           # Raw JSON data
```

### Legacy Result Files
JSON results in `tests/integration/results/`:
- `model_series_YYYYMMDD_HHMMSS.json` - Model series test results
- `batch_results.json` - Batch test results
- `quick_test_YYYYMMDD_HHMMSS.json` - Quick test results

## Notes

1. **Import Paths**: All test files have been updated to use proper import paths from their new locations.

2. **Result Paths**: Test result files are saved to `tests/integration/results/` instead of the previous scattered locations.

3. **No Test Files in Root**: All test files have been moved from the project root to maintain a clean directory structure.

4. **Model Configuration**: Models to test are configured in `model_comparison_config.json`. Available models include gpt-4o, gpt-4o-mini, o3, o3-mini, and o4-mini.

5. **Dependencies**: Some performance tests may install additional dependencies (pandas, tabulate, matplotlib, seaborn) on first run.

6. **Report Generation**: Use `run_model_comparison_with_report.py` for comprehensive HTML reports with visualizations and full LLM outputs.