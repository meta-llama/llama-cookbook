"""
Custom SFT dataset for fine-tuning.
"""

from torchtune.data import OpenAIToMessages
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform


def custom_sft_dataset(
    model_transform: Transform,
    dataset_path: str = "/tmp/train.json",
    train_on_input: bool = False,
    split: str = "train",

) -> SFTDataset:
    """
    Creates a custom SFT dataset for fine-tuning.

    Args:
        dataset_path: Path to the formatted data JSON file
        train_on_input: Whether to train on input tokens
        split: Dataset split to use

    Returns:
        SFTDataset: A dataset ready for fine-tuning with TorchTune
    """
    openaitomessage = OpenAIToMessages(train_on_input=train_on_input)

    ds = SFTDataset(
        source="json",
        data_files=dataset_path,
        split=split,
        message_transform=openaitomessage,
        model_transform=model_transform,
    )
    return ds
