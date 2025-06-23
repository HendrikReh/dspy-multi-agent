# DSPy Multi-Agent System

A production-ready multi-agent system built with DSPy that coordinates research and writing agents to produce comprehensive articles on any topic.

## Features

- **Multi-Agent Architecture**: Coordinated research and writing agents working together
- **Web Search Integration**: Automated research using web search APIs
- **DSPy Framework**: Leveraging DSPy for structured AI workflows
- **FastAPI Integration**: REST API for production deployment
- **Async Support**: Concurrent processing for better performance
- **Type Safety**: Full type hints and mypy validation

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- Optional: Web search API key (for enhanced research capabilities)

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

4. Run the demo:

```bash
python src/main.py
```

### API Server

Start the FastAPI server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

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
SEARCH_API_KEY=your_search_api_key_here  # Optional
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2000
ASYNC_WORKERS=4
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## API Endpoints

### POST /agent/process

Process a research and writing request.

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

### GET /agents/status

Get agent system status and configuration.

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

### Environment Variables

Set required environment variables in your deployment environment:

- `OPENAI_API_KEY` (required)
- `SEARCH_API_KEY` (optional)
- Configure other variables as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and code quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License.
