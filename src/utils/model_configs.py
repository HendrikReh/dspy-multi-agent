# src/utils/model_configs.py
"""Model configurations for o3/o4 variants and testing."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelFamily(Enum):
    """Model family types."""
    O3 = "o3"
    O4 = "o4"
    GPT4 = "gpt-4"  # For current testing


class ModelSize(Enum):
    """Model size variants."""
    FULL = "full"
    MINI = "mini"


class ModelCapability(Enum):
    """Model capability levels for mini variants."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FULL = "full"  # For non-mini models


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""
    name: str
    family: ModelFamily
    size: ModelSize
    capability: ModelCapability
    api_name: str  # Name to use with OpenAI API
    temperature: float = 0.7
    max_tokens: int = 2000
    expected_performance: Dict[str, float] = None  # Expected metrics
    
    def __post_init__(self):
        if self.expected_performance is None:
            self.expected_performance = self._get_default_expectations()
    
    def _get_default_expectations(self) -> Dict[str, float]:
        """Get default performance expectations based on model type."""
        # These are hypothetical values for o3/o4 models
        base_times = {
            ModelFamily.O3: {"first_token": 0.5, "tokens_per_sec": 50},
            ModelFamily.O4: {"first_token": 0.3, "tokens_per_sec": 80},
            ModelFamily.GPT4: {"first_token": 0.8, "tokens_per_sec": 40},
        }
        
        # Adjust for size and capability
        size_multiplier = 1.0 if self.size == ModelSize.FULL else 0.7
        
        capability_multipliers = {
            ModelCapability.LOW: 0.6,
            ModelCapability.MEDIUM: 0.8,
            ModelCapability.HIGH: 0.9,
            ModelCapability.FULL: 1.0,
        }
        
        base = base_times.get(self.family, base_times[ModelFamily.GPT4])
        cap_mult = capability_multipliers[self.capability]
        
        return {
            "expected_first_token": base["first_token"] / (size_multiplier * cap_mult),
            "expected_tokens_per_sec": base["tokens_per_sec"] * size_multiplier * cap_mult,
        }


# Model configurations for testing
MODEL_CONFIGS = {
    # O3 family (with reasoning model requirements)
    "o3": ModelConfig(
        name="o3",
        family=ModelFamily.O3,
        size=ModelSize.FULL,
        capability=ModelCapability.FULL,
        api_name="o3",
        temperature=1.0,  # Required for reasoning models
        max_tokens=25000,  # Required >= 20000 for reasoning models
    ),
    "o3-mini-low": ModelConfig(
        name="o3-mini-low",
        family=ModelFamily.O3,
        size=ModelSize.MINI,
        capability=ModelCapability.LOW,
        api_name="o3-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    "o3-mini-medium": ModelConfig(
        name="o3-mini-medium",
        family=ModelFamily.O3,
        size=ModelSize.MINI,
        capability=ModelCapability.MEDIUM,
        api_name="o3-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    "o3-mini-high": ModelConfig(
        name="o3-mini-high",
        family=ModelFamily.O3,
        size=ModelSize.MINI,
        capability=ModelCapability.HIGH,
        api_name="o3-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    
    # O4 family (with reasoning model requirements)
    "o4": ModelConfig(
        name="o4",
        family=ModelFamily.O4,
        size=ModelSize.FULL,
        capability=ModelCapability.FULL,
        api_name="o4",
        temperature=1.0,  # Required for reasoning models
        max_tokens=25000,  # Required >= 20000 for reasoning models
    ),
    "o4-mini-low": ModelConfig(
        name="o4-mini-low",
        family=ModelFamily.O4,
        size=ModelSize.MINI,
        capability=ModelCapability.LOW,
        api_name="o4-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    "o4-mini-medium": ModelConfig(
        name="o4-mini-medium",
        family=ModelFamily.O4,
        size=ModelSize.MINI,
        capability=ModelCapability.MEDIUM,
        api_name="o4-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    "o4-mini-high": ModelConfig(
        name="o4-mini-high",
        family=ModelFamily.O4,
        size=ModelSize.MINI,
        capability=ModelCapability.HIGH,
        api_name="o4-mini",
        temperature=1.0,  # Required for reasoning models
        max_tokens=20000,  # Minimum for reasoning models
    ),
    
    # Current models for testing (until o3/o4 are available)
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        family=ModelFamily.GPT4,
        size=ModelSize.FULL,
        capability=ModelCapability.FULL,
        api_name="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        family=ModelFamily.GPT4,
        size=ModelSize.MINI,
        capability=ModelCapability.HIGH,
        api_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CONFIGS[model_name]


def get_model_families() -> Dict[ModelFamily, List[str]]:
    """Get models grouped by family."""
    families = {}
    for name, config in MODEL_CONFIGS.items():
        if config.family not in families:
            families[config.family] = []
        families[config.family].append(name)
    return families


def get_test_pairs() -> List[tuple[str, str]]:
    """Get interesting model pairs for comparison."""
    pairs = []
    
    # Compare full models
    pairs.append(("o3", "o4"))
    
    # Compare mini variants within family
    for family in ["o3", "o4"]:
        pairs.extend([
            (f"{family}-mini-low", f"{family}-mini-medium"),
            (f"{family}-mini-medium", f"{family}-mini-high"),
            (f"{family}-mini-low", f"{family}-mini-high"),
        ])
    
    # Compare across families at same capability
    for capability in ["low", "medium", "high"]:
        pairs.append((f"o3-mini-{capability}", f"o4-mini-{capability}"))
    
    # Compare full vs mini-high
    pairs.extend([
        ("o3", "o3-mini-high"),
        ("o4", "o4-mini-high"),
    ])
    
    return pairs


def get_capability_settings(capability: ModelCapability) -> Dict[str, Any]:
    """Get DSPy settings for different capability levels."""
    settings = {
        ModelCapability.LOW: {
            "reasoning_effort": "low",
            "max_reasoning_tokens": 5000,
            "temperature": 0.5,
        },
        ModelCapability.MEDIUM: {
            "reasoning_effort": "medium", 
            "max_reasoning_tokens": 10000,
            "temperature": 0.7,
        },
        ModelCapability.HIGH: {
            "reasoning_effort": "high",
            "max_reasoning_tokens": 20000,
            "temperature": 0.8,
        },
        ModelCapability.FULL: {
            "reasoning_effort": "high",
            "max_reasoning_tokens": 30000,
            "temperature": 0.7,
        },
    }
    return settings.get(capability, settings[ModelCapability.MEDIUM])