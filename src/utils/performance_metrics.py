# src/utils/performance_metrics.py
"""Performance metrics tracking for LLM operations."""
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TokenMetrics:
    """Metrics for token generation."""
    time_to_first_token: float = 0.0
    total_time: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    model_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelComparison:
    """Comparison results between models."""
    model_a_name: str
    model_b_name: str
    model_a_metrics: TokenMetrics
    model_b_metrics: TokenMetrics
    output_similarity: float = 0.0
    content_differences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_a": {
                "name": self.model_a_name,
                "time_to_first_token": self.model_a_metrics.time_to_first_token,
                "total_time": self.model_a_metrics.total_time,
                "tokens_generated": self.model_a_metrics.tokens_generated,
                "tokens_per_second": self.model_a_metrics.tokens_per_second,
            },
            "model_b": {
                "name": self.model_b_name,
                "time_to_first_token": self.model_b_metrics.time_to_first_token,
                "total_time": self.model_b_metrics.total_time,
                "tokens_generated": self.model_b_metrics.tokens_generated,
                "tokens_per_second": self.model_b_metrics.tokens_per_second,
            },
            "comparison": {
                "output_similarity": self.output_similarity,
                "speedup_first_token": self.model_a_metrics.time_to_first_token / self.model_b_metrics.time_to_first_token if self.model_b_metrics.time_to_first_token > 0 else 0,
                "speedup_total": self.model_a_metrics.total_time / self.model_b_metrics.total_time if self.model_b_metrics.total_time > 0 else 0,
                "content_differences": self.content_differences,
            },
            "timestamp": datetime.now().isoformat(),
        }


class PerformanceTracker:
    """Track performance metrics for LLM operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[TokenMetrics]] = {}
        self._start_times: Dict[str, float] = {}
        self._first_token_times: Dict[str, Optional[float]] = {}
        
    def start_tracking(self, operation_id: str, model_name: str) -> None:
        """Start tracking an operation."""
        self._start_times[operation_id] = time.time()
        self._first_token_times[operation_id] = None
        
    def record_first_token(self, operation_id: str) -> None:
        """Record time to first token."""
        if operation_id in self._start_times and self._first_token_times[operation_id] is None:
            self._first_token_times[operation_id] = time.time() - self._start_times[operation_id]
    
    def end_tracking(self, operation_id: str, model_name: str, tokens_generated: int = 0) -> TokenMetrics:
        """End tracking and return metrics."""
        if operation_id not in self._start_times:
            raise ValueError(f"No tracking started for operation {operation_id}")
            
        total_time = time.time() - self._start_times[operation_id]
        time_to_first_token = self._first_token_times.get(operation_id, 0.0) or total_time
        
        metrics = TokenMetrics(
            time_to_first_token=time_to_first_token,
            total_time=total_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            model_name=model_name,
        )
        
        # Store metrics
        if model_name not in self.metrics:
            self.metrics[model_name] = []
        self.metrics[model_name].append(metrics)
        
        # Clean up
        del self._start_times[operation_id]
        if operation_id in self._first_token_times:
            del self._first_token_times[operation_id]
            
        return metrics
    
    def get_average_metrics(self, model_name: str) -> Optional[TokenMetrics]:
        """Get average metrics for a model."""
        if model_name not in self.metrics or not self.metrics[model_name]:
            return None
            
        model_metrics = self.metrics[model_name]
        avg_metrics = TokenMetrics(
            time_to_first_token=sum(m.time_to_first_token for m in model_metrics) / len(model_metrics),
            total_time=sum(m.total_time for m in model_metrics) / len(model_metrics),
            tokens_generated=int(sum(m.tokens_generated for m in model_metrics) / len(model_metrics)),
            tokens_per_second=sum(m.tokens_per_second for m in model_metrics) / len(model_metrics),
            model_name=model_name,
        )
        return avg_metrics
    
    def compare_models(self, model_a: str, model_b: str, output_similarity: float = 0.0) -> ModelComparison:
        """Compare metrics between two models."""
        metrics_a = self.get_average_metrics(model_a)
        metrics_b = self.get_average_metrics(model_b)
        
        if not metrics_a or not metrics_b:
            raise ValueError(f"No metrics available for comparison")
            
        return ModelComparison(
            model_a_name=model_a,
            model_b_name=model_b,
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            output_similarity=output_similarity,
        )


class StreamingMetricsWrapper:
    """Wrapper to track metrics for streaming responses."""
    
    def __init__(self, stream, tracker: PerformanceTracker, operation_id: str, model_name: str):
        self.stream = stream
        self.tracker = tracker
        self.operation_id = operation_id
        self.model_name = model_name
        self.tokens_generated = 0
        self.first_token_recorded = False
        
    async def __aiter__(self):
        """Async iteration with metrics tracking."""
        async for chunk in self.stream:
            if not self.first_token_recorded:
                self.tracker.record_first_token(self.operation_id)
                self.first_token_recorded = True
            
            # Count tokens (simplified - actual implementation would use tiktoken)
            if hasattr(chunk, 'content'):
                self.tokens_generated += len(chunk.content.split())
            
            yield chunk
        
        # End tracking when stream completes
        self.tracker.end_tracking(self.operation_id, self.model_name, self.tokens_generated)


def track_performance(tracker: PerformanceTracker, operation_id: str, model_name: str):
    """Decorator to track performance of async functions."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            tracker.start_tracking(operation_id, model_name)
            try:
                result = await func(*args, **kwargs)
                # Simulate token counting (in real implementation, use tiktoken)
                tokens = len(str(result).split()) if result else 0
                tracker.end_tracking(operation_id, model_name, tokens)
                return result
            except Exception as e:
                tracker.end_tracking(operation_id, model_name, 0)
                raise e
        return wrapper
    return decorator