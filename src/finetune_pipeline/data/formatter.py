import base64
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict, Union


def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


class MessageContent(TypedDict, total=False):
    """Type definition for message content in LLM requests."""

    type: str  # Required field
    text: Optional[str]  # Optional field
    image_url: Optional[Dict[str, str]]  # Optional field


class Message(TypedDict):
    """Type definition for a message in a LLM inference request."""

    role: str
    content: Union[str, List[MessageContent]]


class Conversation:
    """
    Data class representing a conversation which are list of messages.
    """

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []

    def add_message(self, message):
        """
        Add a message to the conversation.

        Args:
            message: Message object to add
        """
        self.messages.append(message)

    def get_message(self, index):
        """
        Get a message at a specific index.

        Args:
            index: Index of the message to retrieve

        Returns:
            Message: The message at the specified index

        Raises:
            IndexError: If the index is out of bounds
        """
        if index < 0 or index >= len(self.messages):
            raise IndexError(
                f"Message index {index} out of range (0-{len(self.messages)-1})"
            )
        return self.messages[index]


class Formatter(ABC):
    """
    Abstract base class for formatters that convert messages to different formats.
    """

    def __init__(self):
        """
        Initialize the formatter.

        Subclasses can override this method to add specific initialization parameters.
        """
        pass

    @abstractmethod
    def format_data(self, data) -> List:
        """
        Format the message. This method must be implemented by subclasses.

        Args:
            data: List of Conversation objects

        Returns:
            List of formatted data
        """
        pass

    @abstractmethod
    def format_conversation(self, conversation) -> Union[Dict, str]:
        """
        Format a sample. This method must be implemented by subclasses.

        Args:
            sample: Conversation object

        Returns:
            Formatted sample in the appropriate format
        """
        pass

    @abstractmethod
    def format_message(self, message) -> Union[Dict, str]:
        """
        Format a message. This method must be implemented by subclasses.

        Args:
            sample: Message object

        Returns:
            Formatted message in the appropriate format
        """
        pass

    # The read_data function has been moved to convert_to_conversations in data_loader.py


class TorchtuneFormatter(Formatter):
    """
    Formatter for Torchtune format.
    """

    def __init__(self):
        """
        Initialize the formatter.
        """
        super().__init__()

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects.

        Returns:
            list: List of formatted data
        """
        if data is None:
            raise ValueError("No data provided to format_data()")

        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_conversation(conversation))
        return formatted_data

    def format_conversation(self, conversation):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            dict: Formatted sample in Torchtune format
        """
        formatted_messages = []
        for message in conversation.messages:
            formatted_messages.append(self.format_message(message))
        return {"messages": formatted_messages}

    def format_message(self, message):
        """
        Format a message in Torchtune format.

        Args:
            message: Message object to format

        Returns:
            dict: Formatted message in Torchtune format
        """
        # For Torchtune format, we can return the Message as is
        # since it's already in a compatible format
        return message


class vLLMFormatter(Formatter):
    """
    Formatter for vLLM format.
    """

    def __init__(self):
        """
        Initialize the formatter.
        """
        super().__init__()

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects.

        Returns:
            list: List of formatted data in vLLM format
        """
        if data is None:
            raise ValueError("No data provided to format_data()")

        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_conversation(conversation))
        return formatted_data

    def format_conversation(self, conversation):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            str: Formatted sample in vLLM format
        """
        formatted_messages = []
        for message in conversation.messages:
            role = message["role"]
            if role == "user":
                formatted_messages.append(self.format_message(message))
        return {"messages": formatted_messages}

    def format_message(self, message):
        """
        Format a message in vLLM format.

        Args:
            message: Message object to format

        Returns:
            str: Formatted message in vLLM format
        """
        contents = []
        vllm_message = {}

        for content in message["content"]:
            if content["type"] == "text":
                contents.append(content["text"])
            elif content["type"] == "image_url":
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


class OpenAIFormatter(Formatter):
    """
    Formatter for OpenAI format.
    """

    def __init__(self):
        """
        Initialize the formatter.
        """
        super().__init__()

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects.

        Returns:
            dict: Formatted data in OpenAI format
        """
        if data is None:
            raise ValueError("No data provided to format_data()")

        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_conversation(conversation))
        return formatted_data

    def format_conversation(self, conversation):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            dict: Formatted sample in OpenAI format
        """
        formatted_messages = []
        for message in conversation.messages:
            formatted_messages.append(self.format_message(message))
        return {"messages": formatted_messages}

    def format_message(self, message):
        """
        Format a message in OpenAI format.

        Args:
            message: Message object to format

        Returns:
            dict: Formatted message in OpenAI format
        """
        # For OpenAI format, we can return the Message as is
        # since it's already in a compatible format
        return message
