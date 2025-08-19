"""
Data processing and formatting type definitions.

Types related to data loading, formatting, and preprocessing.
"""

from typing import Dict, List, Optional, TypedDict, Union


class MessageContent(TypedDict, total=False):
    """Type definition for message content in LLM requests."""

    type: str  # Required field
    text: Optional[str]  # Optional field
    image_url: Optional[Dict[str, str]]  # Optional field


class Message(TypedDict):
    """Type definition for a message in a LLM inference request."""

    role: str
    content: Union[str, List[MessageContent]]


class DataLoaderConfig(TypedDict, total=False):
    """Configuration for data loading."""

    batch_size: int
    shuffle: bool
    data_path: str
    validation_split: Optional[float]


class FormatterConfig(TypedDict, total=False):
    """Configuration for data formatting."""

    format_type: str  # "torchtune", "vllm", "openai"
    include_system_prompt: bool
    max_sequence_length: Optional[int]


class DatasetStats(TypedDict):
    """Statistics about a dataset."""

    total_conversations: int
    total_messages: int
    avg_messages_per_conversation: float
    data_size_mb: float
