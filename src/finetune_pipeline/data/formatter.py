from types import Message
from typing import Callable, List, Union


def format_huggingface(base_message: Message) -> Message:
    if base_message["role"] != "user":
        return base_message

    contents = []
    for content in base_message["content"]:
        if content["type"] == "text":
            contents.append(content)
        elif content["type"] == "image_url":
            contents.append({"type": "image", "url": content["image_url"]["url"]})

    return {"role": "user", "content": contents}


def apply_format(
    data: Union[List[Message], List[List[Message]]], format_func: Callable
):
    """
    Apply the format function to the data.

    Args:
        data: Either a list of message dictionaries or a list of conversations
              (where each conversation is a list of message dictionaries)
        format_func: Function that formats a single message dictionary

    Returns:
        List of formatted dictionaries
    """
    if not data:
        return []

    if isinstance(data[0], Message):
        return [format_func(message) for message in data]

    if isinstance(data[0][0], Message):
        return [apply_format(conversation, format_func) for conversation in data]

    raise ValueError("Invalid data format")
