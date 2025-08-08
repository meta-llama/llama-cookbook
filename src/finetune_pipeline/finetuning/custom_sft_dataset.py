"""
Custom SFT dataset for fine-tuning.
"""
from typing import Any, List, Mapping
from torchtune.data import OpenAIToMessages
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform
from torchtune.data import load_image, Message

class MessageTransform(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:

        user_content = []
        assistant_content = []

        for message in sample["messages"]:
            contents = message['content']
            role = message['role']
            for content in contents:
                typ = content['type']
                val = None
                if typ == 'text':
                    val = content['text']
                    if role == 'user':
                        user_content.append({"type": "text", "content": val})
                    else:
                        assistant_content.append({"type": "text", "content": val})
                elif typ == 'image' or typ == 'image_url':
                    val = load_image(content['image'])
                    user_content.append({"type": "image", "content": val})

        messages = [
            Message(
                role="user",
                content=user_content,
                masked=True,
                eot=True,
            ),
            Message(
                role="assistant",
                content=assistant_content,
                masked=False,
                eot=True,
            ),
        ]

        return {"messages": messages}


def custom_sft_dataset(
    model_transform: Transform,
    *,
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
    # message_transform = OpenAIToMessages(train_on_input=train_on_input)
    message_transform = MessageTransform()

    ds = SFTDataset(
        source="json",
        data_files="/home/ubuntu/yash-workspace/outputs/train_torchtune_formatted_data.json",
        split="train",
        message_transform=message_transform,
        model_transform=model_transform,
    )
    return ds
