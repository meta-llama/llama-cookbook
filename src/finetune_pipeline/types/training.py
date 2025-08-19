"""
Training and finetuning type definitions.

Types related to model training, finetuning configurations, and training metrics.
"""

from typing import List, Optional, TypedDict


class TrainingConfig(TypedDict, total=False):
    """Configuration for training parameters."""

    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str
    optimizer: Optional[str]
    weight_decay: Optional[float]
    gradient_accumulation_steps: Optional[int]


class LoRAConfig(TypedDict, total=False):
    """Configuration for LoRA (Low-Rank Adaptation) finetuning."""

    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]


class TrainingMetrics(TypedDict):
    """Training metrics collected during training."""

    epoch: int
    step: int
    train_loss: float
    validation_loss: Optional[float]
    learning_rate: float
    throughput_tokens_per_sec: float


class CheckpointInfo(TypedDict):
    """Information about a model checkpoint."""

    checkpoint_path: str
    epoch: int
    step: int
    model_name: str
    training_config: TrainingConfig
    metrics: TrainingMetrics
