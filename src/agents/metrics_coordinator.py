# src/agents/metrics_coordinator.py
"""Enhanced coordinator with performance metrics tracking."""
import time
import asyncio
from typing import Dict, Any, Optional
from agents.coordinator import MultiAgentCoordinator
from utils.performance_metrics import PerformanceTracker, TokenMetrics
import dspy


class MetricsCoordinator(MultiAgentCoordinator):
    """Multi-agent coordinator with performance metrics tracking."""
    
    def __init__(self, tavily_api_key: Optional[str] = None, tracker: Optional[PerformanceTracker] = None):
        super().__init__(tavily_api_key)
        self.tracker = tracker or PerformanceTracker()
        self.current_model = None
        
    def set_model(self, model_name: str) -> None:
        """Set the current model being used."""
        self.current_model = model_name
        
    async def forward(
        self, request: str, target_audience: str = "general", max_sources: int = 5
    ) -> Dict[str, Any]:
        """Execute workflow with metrics tracking."""
        if not self.current_model:
            raise ValueError("Model name must be set before calling forward")
            
        operation_id = f"{self.current_model}_{time.time()}"
        
        # Start tracking
        self.tracker.start_tracking(operation_id, self.current_model)
        
        try:
            # Track task planning
            plan_start = time.time()
            plan = self.task_planner(request=request)
            plan_time = time.time() - plan_start
            
            # Record first token (simulated - in reality would track actual LLM response)
            self.tracker.record_first_token(operation_id)
            
            # Extract topic
            topic = request
            
            # Phase 1: Research with timing
            research_start = time.time()
            research_results = await self.researcher.forward(
                topic=topic, max_sources=max_sources
            )
            research_time = time.time() - research_start
            
            # Phase 2: Writing with timing
            writing_start = time.time()
            writing_results = self.writer.forward(
                topic=topic,
                research_data=research_results["research_results"],
                key_points=research_results["key_points"],
                target_audience=target_audience,
            )
            writing_time = time.time() - writing_start
            
            # Calculate total tokens (simplified)
            total_text = str(research_results) + str(writing_results)
            tokens_generated = len(total_text.split())
            
            # End tracking
            metrics = self.tracker.end_tracking(operation_id, self.current_model, tokens_generated)
            
            # Combine results with timing info
            return {
                "request": request,
                "topic": topic,
                "research_phase": research_results,
                "writing_phase": writing_results,
                "final_article": writing_results["article"],
                "summary": writing_results["summary"],
                "sources": research_results["sources"],
                "metrics": {
                    "model": self.current_model,
                    "total_time": metrics.total_time,
                    "time_to_first_token": metrics.time_to_first_token,
                    "tokens_generated": metrics.tokens_generated,
                    "tokens_per_second": metrics.tokens_per_second,
                    "phase_timings": {
                        "planning": plan_time,
                        "research": research_time,
                        "writing": writing_time,
                    },
                },
            }
            
        except Exception as e:
            # End tracking even on error
            self.tracker.end_tracking(operation_id, self.current_model, 0)
            raise e