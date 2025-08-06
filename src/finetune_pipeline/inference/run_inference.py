"""
Script to run the model inference.

"""

from typing import Any, Dict, List, Optional, TypedDict, Union

from vllm import SamplingParams


class InferenceRequest(TypedDict, total=False):
    """Type definition for LLM inference request."""

    model: str
    messages: List[Message]
    temperature: float
    top_p: float
    max_completion_tokens: int
    seed: int
    response_format: Optional[Dict[str, Any]]


class VLLMInferenceRequest(TypedDict):
    """Type definition for VLLM inference request format."""

    messages: List[List[Message]]
    sampling_params: Union[SamplingParams, List[SamplingParams]]
