# DSPy Multi-Agent System

A multi-agent system built with DSPy that coordinates research and writing agents to produce comprehensive articles on any topic.

## Features

- **Multi-Agent Architecture**: Coordinated research and writing agents working together
- **Web Search Integration**: Automated research using Tavily search API
- **DSPy Framework**: Leveraging DSPy for structured AI workflows
- **FastAPI Integration**: REST API for production deployment
- **Async Support**: Concurrent processing for better performance
- **Type Safety**: Full type hints and mypy validation
- **Comprehensive error handling and resource management**

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- Tavily API key (for web search functionality) - Get one at https://tavily.com

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd dspy-multi-agent
```

2. Install dependencies using uv:

```bash
uv sync --dev
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the System

#### Option 1: Command Line Demo

```bash
uv run python src/main.py
```

#### Option 2: FastAPI Server + Demo Client

**Start the server:**

```bash
python start_api.py
# or
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Run the demo via API (2 methods):**

**Method 1 - Python client (recommended):**

```bash
uv run python test/demo_client.py
```

**Method 2 - Direct curl command:**

```bash
curl -X POST http://localhost:8000/agent/demo
```

#### Option 3: Custom API Requests

**FastAPI Server**

**Method 1 - Direct command:**

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Method 2 - Startup script (recommended):**

```bash
python start_api.py
```

The API will be available at:

- **Main API**: <http://localhost:8000>
- **Interactive docs**: <http://localhost:8000/docs>
- **Alternative docs**: <http://localhost:8000/redoc>

## Architecture

### Core Components

- **MultiAgentCoordinator**: Orchestrates the multi-agent workflow
- **ResearchAgent**: Conducts web research and information gathering
- **WriterAgent**: Creates and polishes articles based on research
- **WebSearchTool**: Handles web search API integration

### Workflow

1. **Task Planning**: Coordinator analyzes the request and creates execution plan
2. **Research Phase**: ResearchAgent searches for information and extracts key findings
3. **Writing Phase**: WriterAgent creates draft article and polishes it
4. **Result Compilation**: Final article with sources and summary

## Configuration

Environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Required for web search
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2000
ASYNC_WORKERS=4
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## API Endpoints

### POST /agent/demo

Run the built-in demo request (same as `src/main.py`).

**No request body required.**

**Response:**

```json
{
  "status": "success",
  "topic": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
  "article": "Full article content...",
  "summary": "Article summary...",
  "sources": ["source1", "source2", "..."],
  "key_points": ["point1", "point2", "..."],
  "processing_time": 47.2,
  "agent_id": "uuid"
}
```

### POST /agent/process

Process a custom research and writing request.

**Request Body:**

```json
{
  "query": "Write about artificial intelligence in healthcare",
  "target_audience": "healthcare professionals",
  "max_sources": 10
}
```

**Response:**

```json
{
  "status": "success",
  "topic": "artificial intelligence in healthcare",
  "article": "Full article content...",
  "summary": "Article summary...",
  "sources": ["source1", "source2", "..."],
  "key_points": ["point1", "point2", "..."],
  "processing_time": 15.2,
  "agent_id": "uuid"
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-23T14:35:10.956221",
  "version": "1.0.0"
}
```

### GET /agents/status

Get agent system status and configuration.

**Response:**

```json
{
  "coordinator_ready": true,
  "model_configured": true,
  "async_workers": 4
}
```

## Development

### Code Quality

Format code:

```bash
black .
```

Type checking:

```bash
mypy src/
```

Run tests:

```bash
uv run pytest
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Testing the API

### Demo Request (Built-in)

```bash
# Python client (recommended)
uv run python test/demo_client.py

# Or curl command
curl -X POST "http://localhost:8000/agent/demo"
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### Custom Process Request

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Write about AI in healthcare",
       "target_audience": "healthcare professionals",
       "max_sources": 5
     }'
```

### Agent Status

```bash
curl -X GET "http://localhost:8000/agents/status"
```

## Extending the System

### Adding New Agents

1. Create agent class inheriting from `dspy.Module`
2. Define agent signature in `src/signatures/agent_signatures.py`
3. Implement `forward` method with your agent logic
4. Register agent in coordinator

### Custom Search Integration

Replace the placeholder search API in `WebSearchTool.search()` with your preferred search service:

- You.com Search API
- Serper.dev
- Bing Search API
- Custom search implementation

### Adding Memory/Persistence

Agents can be extended with:

- Vector databases for long-term memory
- Session storage for conversation context
- Database integration for result persistence

## Production Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup

Ensure all required environment variables are set in production:

```bash
export OPENAI_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"  # Required for web search
export MODEL_NAME="gpt-4o-mini"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## Model Comparison and Testing

### Running Model Comparisons

Compare multiple AI models with comprehensive reporting:

```bash
# Run with default configuration (models from model_comparison_config.json)
uv run python tests/models_batch_run/run_model_comparison_with_report.py

# Use custom configuration file
uv run python tests/models_batch_run/run_model_comparison_with_report.py --config custom_config.json

# Override specific parameters
uv run python tests/models_batch_run/run_model_comparison_with_report.py --models gpt-4o o3-mini --query "Explain quantum computing"
```

### Model Configuration

Models are configured in `model_comparison_config.json`:

```json
{
  "models": ["gpt-4o-mini"],  // Models to compare
  "default_query": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
  "default_audience": "general audience",
  "default_sources": 5
}
```

### Test Reports

Model comparison tests generate comprehensive HTML reports in the `reports/` directory including:
- Performance metrics and visualizations (performance charts, heatmaps, speedup matrices)
- Token usage and speed comparisons
- Side-by-side output comparisons with similarity metrics
- Full LLM outputs for each model (actual responses, not placeholders)
- Detailed phase timing breakdowns (planning, research, writing)
- Cross-model similarity analysis with key points extraction

Report structure:
```
reports/
└── model_comparison_YYYYMMDD_HHMMSS/
    ├── report_YYYYMMDD_HHMMSS.html     # Main comprehensive report
    ├── llm_outputs/                     # Raw model outputs
    ├── visualizations/                  # Generated charts and graphs
    └── data/                           # JSON data files
```

## Recent Updates

- ✅ Fixed all import errors and path resolution issues
- ✅ Added comprehensive type annotations (mypy compliant)
- ✅ Implemented proper async resource management
- ✅ Fixed FastAPI server startup issues
- ✅ Added startup script for easier development
- ✅ Added model comparison framework with comprehensive reporting
- ✅ External configuration file support for model testing
- ✅ Fixed LLM output capture to save actual model responses

## License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007 (see LICENSE)

## Contributing

[Add contribution guidelines here]

## Visualization Tools

The project includes tools for visualizing test results:

```bash
# Generate visualizations from test results
uv run python tests/tools/visualize_results.py

# Compare specific model results
uv run python tests/tools/visualize_comparison_results.py
```

## Documentation

- **API Testing Guide**: [docs/API Testing Examples.md](docs/API%20Testing%20Examples.md) - Comprehensive testing examples and curl commands
- **Model Comparison Guide**: [docs/model_comparison.md](docs/model_comparison.md) - Detailed guide for model testing
- **Test Organization**: [docs/test_organization.md](docs/test_organization.md) - Test structure and usage
- **Visualization Guide**: [docs/visualization_guide.md](docs/visualization_guide.md) - Using visualization tools
- **Report Generation Guide**: [docs/comprehensive_report_guide.md](docs/comprehensive_report_guide.md) - Understanding test reports
- **Interactive API Docs**: <http://localhost:8000/docs> (when server is running)
- **Alternative API Docs**: <http://localhost:8000/redoc> (when server is running)
