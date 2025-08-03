# src/utils/config.py
import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration manager for DSPy multi-agent system."""

    def __init__(self) -> None:
        self.load_environment_variables()
        self.validate_config()

    def load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        self.async_workers = int(os.getenv("ASYNC_WORKERS", "4"))

        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def validate_config(self) -> None:
        """Validate required configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if not self.tavily_api_key:
            print("Warning: TAVILY_API_KEY not configured. Web search functionality will be disabled.")
