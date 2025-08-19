"""
Inference and serving type definitions.

Types related to model inference, serving configurations, and inference results.
"""

from typing import Dict, List, Optional, TypedDict


class InferenceConfig(TypedDict, total=False):
    """Configuration for inference parameters."""

    model_path: str
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    repetition_penalty: Optional[float]


class InferenceRequest(TypedDict):
    """Request for model inference."""

    messages: List[Dict]  # List of Message objects
    config: InferenceConfig
    stream: Optional[bool]


class InferenceResponse(TypedDict):
    """Response from model inference."""

    generated_text: str
    input_tokens: int
    output_tokens: int
    total_time_ms: float
    tokens_per_second: float


class ModelServingConfig(TypedDict, total=False):
    """Configuration for model serving."""

    host: str
    port: int
    model_path: str
    max_concurrent_requests: int
    gpu_memory_utilization: Optional[float]
    tensor_parallel_size: Optional[int]


class ServingMetrics(TypedDict):
    """Metrics for model serving."""

    requests_per_second: float
    average_latency_ms: float
    active_requests: int
    total_requests: int
    error_rate: float
