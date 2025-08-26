from typing import Dict, List, Union


def format_message_torchtune(message: Dict) -> Dict:
    """Format a message in Torchtune format."""
    return message


def format_message_openai(message: Dict) -> Dict:
    """Format a message in OpenAI format."""
    contents = []
    for content in message["content"]:
        if content["type"] == "text":
            contents.append({"type": "input_text", "text": content["text"]})
        elif content["type"] == "image_url":
            contents.append(
                {"type": "input_image", "image_url": content["image_url"]["url"]}
            )
        else:
            raise ValueError(f"Unknown content type: {content['type']}")
    return {"role": message["role"], "content": contents}


def format_message_vllm(message: Dict) -> Dict:
    """Format a message in vLLM format."""
    contents = []
    vllm_message = {}

    for content in message["content"]:
        if content["type"] == "text":
            contents.append(content)
        elif content["type"] == "image_url" or content["type"] == "image":
            img_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{content["image_url"]["url"]}"
                },
            }
            contents.append(img_content)
        else:
            raise ValueError(f"Unknown content type: {content['type']}")
    vllm_message["role"] = message["role"]
    vllm_message["content"] = contents
    return vllm_message


def apply_format(data: Union[List[Dict], List[List[Dict]]], format_func) -> List[Dict]:
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

    # Check if data is a list of conversations (list of lists) or a list of messages
    if isinstance(data[0], list):
        # data is a list of conversations, each conversation is a list of messages
        formatted_conversations = []
        for conversation in data:
            formatted_messages = []
            for message in conversation:
                formatted_message = format_func(message)
                formatted_messages.append(formatted_message)
            # Return the conversation as a dictionary with "messages" key
            formatted_conversations.append({"messages": formatted_messages})
        return formatted_conversations
    else:
        # data is a list of messages
        formatted_messages = []
        for message in data:
            formatted_message = format_func(message)
            formatted_messages.append(formatted_message)
        return formatted_messages
