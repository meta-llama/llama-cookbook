from pprint import pprint
from typing import Any, Mapping

from torchtune.data import Message
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class ToolCallMessages(Transform):
    def __init__(self, train_on_input=False):
        self._role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "tool": "ipython",
            "user": "user",
            "assistant": "assistant",
        }
        self.train_on_input = train_on_input

    def __call__(self, sample):
        conversations = sample["cot_conversations"]
        messages = []

        # Keep the original list comprehension structure but add the EOT logic
        for i, msg in enumerate(conversations):
            next_is_tool = (
                i < len(conversations) - 1 and conversations[i + 1]["from"] == "tool"
            )

            messages.append(
                Message(
                    role=self._role_map[msg["from"]],
                    content=msg["value"],
                    masked=(
                        False
                        if self.train_on_input
                        else self._role_map[msg["from"]] != "assistant"
                    ),
                    eot=not (
                        msg["from"] == "tool" or (msg["from"] == "gpt" and next_is_tool)
                    ),
                )
            )
        return {"messages": messages}


def custom_dataset(
    model_transform, train_on_input=False, **load_dataset_kwargs
) -> SFTDataset:
    message_transform = ToolCallMessages(train_on_input=train_on_input)
    return SFTDataset(
        source="json",
        data_files="train_final_mix.json",
        split="train",
        message_transform=message_transform,
        model_transform=model_transform,
        **load_dataset_kwargs,
    )
