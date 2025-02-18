from pprint import pprint
from typing import Any, Mapping

from torchtune.data import _messages, Message
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer
class ToolCallMessages(Transform):
    def __init__(self):
        self._role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "tool": "ipython",
        }

    def __call__(self, sample):
        messages = [
            Message(
                role=self._role_map[msg["from"]],
                content=msg["value"],
                masked=self._role_map[msg["from"]] != "assistant",
                eot=True,
            )
            for msg in sample["cot_conversations"]
        ]
        return {"messages": messages}


def custom_dataset(model_transform, **load_dataset_kwargs) -> SFTDataset:
    message_transform = ToolCallMessages()
    return SFTDataset(
        source="json",
        data_files="train_data.json",
        split="train",
        message_transform=message_transform,
        model_transform=ModelTokenizer,
        **load_dataset_kwargs,
    )
