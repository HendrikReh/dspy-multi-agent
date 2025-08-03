# src/api/main.py
import time
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import dspy

# Fix imports to work with uvicorn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.coordinator import MultiAgentCoordinator
from api.models import AgentRequest, AgentResponse, HealthResponse
from utils.config import Config
from utils.error_handling import setup_logging


# Global variables
coordinator: Optional[MultiAgentCoordinator] = None
config: Optional[Config] = None
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global coordinator, config, logger

    # Startup
    config = Config()
    
    # Setup logging (same as main.py)
    logger = setup_logging(config.log_level)
    logger.info("FastAPI server starting up...")

    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini", api_key=config.openai_api_key, temperature=0.7
    )
    dspy.configure(lm=lm, async_max_workers=4)

    # Initialize coordinator with Tavily API key
    coordinator = MultiAgentCoordinator(tavily_api_key=config.tavily_api_key)
    
    logger.info("FastAPI server startup complete")

    yield

    # Shutdown
    logger.info("FastAPI server shutting down...")
    # Clean up resources if needed


app = FastAPI(
    title="DSPy Multi-Agent API",
    description="Production-ready DSPy multi-agent system for research and writing",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agent/demo", response_model=AgentResponse)
async def demo_request() -> AgentResponse:
    """Run the demo request from main.py - comprehensive AI healthcare article."""
    try:
        start_time = time.time()
        agent_id = str(uuid.uuid4())

        # Demo request (same as in main.py)
        demo_query = "Write a comprehensive article about the impact of artificial intelligence on modern healthcare"
        demo_target_audience = "healthcare professionals"
        demo_max_sources = 10

        # Process with multi-agent system
        if coordinator is None:
            raise HTTPException(status_code=500, detail="Coordinator not initialized")
            
        result = await coordinator.forward(
            request=demo_query,
            target_audience=demo_target_audience,
            max_sources=demo_max_sources,
        )

        processing_time = time.time() - start_time

        return AgentResponse(
            status="success",
            topic=result["topic"],
            article=result["final_article"],
            summary=result["summary"],
            sources=result["sources"],
            key_points=result["research_phase"]["key_points"],
            processing_time=processing_time,
            agent_id=agent_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/process", response_model=AgentResponse)
async def process_request(request: AgentRequest) -> AgentResponse:
    """Process a research and writing request."""
    try:
        start_time = time.time()
        agent_id = str(uuid.uuid4())

        # Process with multi-agent system
        if coordinator is None:
            raise HTTPException(status_code=500, detail="Coordinator not initialized")
            
        result = await coordinator.forward(
            request=request.query,
            target_audience=request.target_audience,
            max_sources=request.max_sources,
        )

        processing_time = time.time() - start_time

        return AgentResponse(
            status="success",
            topic=result["topic"],
            article=result["final_article"],
            summary=result["summary"],
            sources=result["sources"],
            key_points=result["research_phase"]["key_points"],
            processing_time=processing_time,
            agent_id=agent_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now(), version="1.0.0")


@app.get("/agents/status")
async def agent_status() -> Dict[str, Any]:
    """Get agent system status."""
    return {
        "coordinator_ready": coordinator is not None,
        "model_configured": dspy.settings.lm is not None,
        "async_workers": dspy.settings.async_max_workers,
    }
