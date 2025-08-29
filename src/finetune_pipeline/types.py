"""
Core type definitions for the finetune pipeline.

This module contains TypedDicts and type definitions that are used across
multiple modules in the pipeline.
"""

from typing import Dict, List, Optional, TypedDict, Union


class MessageContent(TypedDict, total=False):
    """Type definition for message content in LLM requests."""

    type: str  # "text", "image_url""
    text: Optional[str]  # Optional field
    image_url: Optional[Dict[str, str]]  # Optional field


class Message(TypedDict):
    """Type definition for a message in a LLM inference request."""

    role: str
    content: Union[str, List[MessageContent]]


class TrainingConfig(TypedDict, total=False):
    """Configuration for training parameters."""

    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str
    optimizer: Optional[str]


class InferenceConfig(TypedDict, total=False):
    """Configuration for inference parameters."""

    model_path: str
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
