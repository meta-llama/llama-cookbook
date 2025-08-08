from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform
from torchtune.data import OpenAIToMessages


def custom_sft_dataset(
    model_transform: Transform,
    *,
    split: str = "train",
    dataset_path: str = "files/synthetic_data/train.csv",
    train_on_input: bool = True,
) -> SFTDataset:
    """Creates a custom dataset."""

    openaitomessage = OpenAIToMessages(train_on_input=train_on_input)

    ds = SFTDataset(
        source="json",
        data_files=dataset_path,
        split="train",
        message_transform=openaitomessage,
        model_transform=Transform,
    )
    return ds
