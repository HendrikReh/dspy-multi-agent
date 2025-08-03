# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent system built with DSPy that coordinates research and writing agents to produce comprehensive articles. The system uses FastAPI for production deployment and integrates with OpenAI's API and optional web search APIs.

## Development Commands

### Running the System

```bash
# Install dependencies with uv
uv sync --dev

# Command line demo
uv run python src/main.py

# Start FastAPI server (two methods)
python start_api.py
# or
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run demo client against API
uv run python test/demo_client.py

# Run model comparison with report generation
uv run python tests/models_batch_run/run_model_comparison_with_report.py

# Run with custom configuration file
uv run python tests/models_batch_run/run_model_comparison_with_report.py --config custom_config.json
```

### Code Quality

```bash
# Format code
black .

# Type checking
mypy src/

# Run tests
uv run pytest

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Architecture

The system follows a multi-agent architecture with three main components:

1. **MultiAgentCoordinator** (`src/agents/coordinator.py`): Orchestrates the workflow between research and writing agents. Handles task planning and result compilation.

2. **ResearchAgent** (`src/agents/researcher.py`): Conducts web research using the WebSearchTool. Extracts key findings and sources.

3. **WriterAgent** (`src/agents/writer.py`): Creates and polishes articles based on research data. Handles draft generation and refinement.

The API layer (`src/api/`) provides REST endpoints for production deployment with FastAPI. The coordinator is initialized as a global singleton with proper async resource management.

## Key Implementation Details

- **DSPy Configuration**: The system uses `dspy.LM` with OpenAI's GPT-4o-mini model. Configuration happens in both `src/main.py` and `src/api/main.py`.

- **Async Architecture**: The coordinator and research agent use async/await for concurrent processing. The API uses FastAPI's lifespan context manager for proper startup/shutdown.

- **Import Path Resolution**: The API adds the parent directory to sys.path to resolve imports when running with uvicorn.

- **Environment Variables**: Configuration is centralized in `src/utils/config.py` which loads from environment variables or `.env` file.

- **Error Handling**: Comprehensive error handling with proper logging setup in `src/utils/error_handling.py`.

## API Endpoints

- `POST /agent/demo`: Run the built-in demo (same as src/main.py)
- `POST /agent/process`: Process custom research/writing requests
- `GET /health`: Health check
- `GET /agents/status`: Agent system status

## Testing

When making changes, ensure to:
1. Run `mypy src/` to verify type safety
2. Format code with `black .`
3. Run unit tests: `uv run pytest tests/unit/`
4. Test the API with the demo client: `uv run python demo_client.py`
5. Check API docs at http://localhost:8000/docs when server is running

### Test Organization

All test files are organized under the `tests/` directory:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests and test results
- `tests/performance/` - Performance and model comparison tests
- `tests/tools/` - Testing utilities like visualization
- `tests/models_batch_run/` - Model comparison batch scripts

See `docs/test_organization.md` for detailed test structure and usage.

### Visualization Tools

```bash
# Generate performance visualizations from test results
uv run python tests/tools/visualize_results.py

# Compare specific model results
uv run python tests/tools/visualize_comparison_results.py
```

## Model Comparison Testing

### Configuration

Model comparison tests are configured via `model_comparison_config.json`:
- Defines which models to test (currently set to `["gpt-4o-mini"]`)
- Default query, audience, and source settings
- Can be overridden with command-line arguments or custom config files

### Running Model Comparisons

```bash
# Use default configuration
uv run python tests/models_batch_run/run_model_comparison_with_report.py

# Custom configuration file
uv run python tests/models_batch_run/run_model_comparison_with_report.py --config custom_config.json

# Override specific settings
uv run python tests/models_batch_run/run_model_comparison_with_report.py --models gpt-4o o3 --query "Custom query"
```

### Test Reports

Model comparison generates comprehensive HTML reports in `reports/` with:
- Performance metrics and visualizations (charts, heatmaps, speedup matrices)
- Token usage analysis with per-second throughput
- Side-by-side output comparisons with similarity metrics
- Full LLM outputs (actual model responses, not placeholders)
- Phase-by-phase timing breakdowns (planning, research, writing)
- Cross-model similarity analysis with key points extraction

Report structure:
```
reports/
└── model_comparison_YYYYMMDD_HHMMSS/
    ├── report_YYYYMMDD_HHMMSS.html     # Main comprehensive report
    ├── llm_outputs/                     # Raw model outputs
    ├── visualizations/                  # Generated charts (PNG files)
    └── data/                           # JSON data files
```

### Recent Improvements

- Fixed LLM output capture to save actual model responses instead of placeholder text
- Added external configuration file support for flexible model testing
- Improved report generation with better visualization and metrics
- Added comprehensive report generator with multiple visualization types
- Fixed import paths and parameter passing in test framework