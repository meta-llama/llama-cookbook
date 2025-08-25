import base64
from typing import Dict, List


def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def format_message_vllm(message: Dict) -> Dict:
    """Format a message in vLLM format."""
    contents = []
    vllm_message = {}

    for content in message["content"]:
        if content["type"] == "text":
            contents.append(content)
        elif content["type"] == "image_url" or content["type"] == "image":
            base64_image = image_to_base64(content["image_url"]["url"])
            img_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
            }
            contents.append(img_content)
        else:
            raise ValueError(f"Unknown content type: {content['type']}")
    vllm_message["role"] = message["role"]
    vllm_message["content"] = contents
    return vllm_message


def format_conversation_vllm(conversation) -> Dict:
    """Format a conversation in vLLM format."""
    formatted_messages = []
    for message in conversation.messages:
        role = message["role"]
        if role != "assistant":
            formatted_messages.append(format_message_vllm(message))
    return {"messages": formatted_messages}


# TODO: Remove
def format_conversation_openai(conversation) -> Dict:
    """Format a conversation in OpenAI format."""
    formatted_messages = []
    for message in conversation.messages:
        formatted_messages.append(format_message_openai(message))
    return {"messages": formatted_messages}


# TODO: Remove
def format_data_torchtune(data: List[Conversation]) -> List[Dict]:
    """Format data in Torchtune format."""
    if data is None:
        raise ValueError("No data provided to format_data()")

    return [format_conversation_torchtune(conversation) for conversation in data]


def format_data_vllm(data: List[Conversation]) -> List[Dict]:
    """Format data in vLLM format."""
    if data is None:
        raise ValueError("No data provided to format_data()")

    return [format_conversation_vllm(conversation) for conversation in data]


# TODO: Remove
def format_data_openai(data: List[Conversation]) -> List[Dict]:
    """Format data in OpenAI format."""
    if data is None:
        raise ValueError("No data provided to format_data()")

    return [format_conversation_openai(conversation) for conversation in data]


# Dictionary to map format names to functions for easy dispatch
FORMATTERS = {
    "torchtune": {
        "data": format_data_torchtune,
        "conversation": format_conversation_torchtune,
        "message": format_message_torchtune,
    },
    "vllm": {
        "data": format_data_vllm,
        "conversation": format_conversation_vllm,
        "message": format_message_vllm,
    },
    "openai": {
        "data": format_data_openai,
        "conversation": format_conversation_openai,
        "message": format_message_openai,
    },
}


def format_data(data: List[Conversation], format_type: str) -> List[Dict]:
    """
    Generic function to format data in the specified format.

    Args:
        data: List of Conversation objects
        format_type: One of "torchtune", "vllm", "openai"

    Returns:
        List of formatted data
    """
    if format_type not in FORMATTERS:
        raise ValueError(
            f"Unknown format type: {format_type}. Supported: {list(FORMATTERS.keys())}"
        )

    return FORMATTERS[format_type]["data"](data)
