# src/utils/error_handling.py
import logging
import time
import asyncio
from functools import wraps
from typing import Callable, Any, Optional
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
            return None

        return wrapper

    return decorator


# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("dspy_multi_agent.log")],
    )

    return logging.getLogger(__name__)
