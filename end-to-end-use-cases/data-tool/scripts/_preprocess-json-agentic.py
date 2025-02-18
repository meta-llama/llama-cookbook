import json

from datasets import DatasetDict, load_from_disk


def preprocess_conversation(example):
    for conv in example["conversations"]:
        if conv["from"] == "human":
            conv["from"] = "user"
        elif conv["from"] == "gpt":
            conv["from"] = "assistant"
    return example


def transform_conversations(example):
    """Transform conversations list to string format."""
    conv_str = "\n".join(
        f"{msg['from']}: {msg['value']}" for msg in example["conversations"]
    )
    return {"id": example["id"], "conversations": conv_str}

json_balanced = load_from_disk("balanced-json-modeagentic")

processed_dataset = json_balanced.map(preprocess_conversation)

processed_dataset = processed_dataset["train"].map(
    transform_conversations, remove_columns=["category", "subcategory", "schema"]
)

processed_dataset.save_to_disk("json-agentic-balanced-final")
