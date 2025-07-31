from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict, Union


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
    def format_sample(self, sample) -> Union[Dict, str]:
        """
        Format a sample. This method must be implemented by subclasses.

        Args:
            sample: Conversation object

        Returns:
            Formatted sample in the appropriate format
        """
        pass

    def read_data(self, data):
        """
        Read Hugging Face data and convert it to a list of Conversation objects.

        Args:
            data: Hugging Face dataset or iterable data

        Returns:
            list: List of Conversation objects, each containing a list of Message objects
        """
        conversations = []
        for item in data:
            # Extract fields from the Hugging Face dataset item
            image = item.get("image", None)
            input_text = item.get("input", "")
            output_label = item.get("output", "")

            # Create a new conversation
            conversation = Conversation()

            # Create user content and user message
            user_content = [
                {"type": "text", "text": input_text},
            ]
            # Add image to user content
            if image is not None:
                user_content.append({"type": "image", "image_url": {"url": image}})

            user_message = {"role": "user", "content": user_content}

            # Create assistant message with text content
            assistant_content = [
                {"type": "text", "text": output_label},
            ]
            assistant_message = {"role": "assistant", "content": assistant_content}

            # Add messages to the conversation
            conversation.add_message(user_message)
            conversation.add_message(assistant_message)

            # Add the conversation to the list
            conversations.append(conversation)

        return conversations


class TorchtuneFormatter(Formatter):
    """
    Formatter for Torchtune format.
    """

    def __init__(self, data=None):
        """
        Initialize the formatter.

        Args:
            data: Optional data to initialize with
        """
        super().__init__()
        self.conversation_data = None
        if data is not None:
            self.conversation_data = self.read_data(data)

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects

        Returns:
            list: List of formatted data
        """
        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_sample(conversation))
        return formatted_data

    def format_sample(self, sample):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            dict: Formatted sample in Torchtune format
        """
        formatted_messages = []
        for message in sample.messages:
            formatted_messages.append(self.format(message))
        return {"messages": formatted_messages}

    def format(self, message):
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

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects

        Returns:
            list: List of formatted data in vLLM format
        """
        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_sample(conversation))
        return formatted_data

    def format_sample(self, sample):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            str: Formatted sample in vLLM format
        """
        formatted_messages = []
        for message in sample.messages:
            formatted_messages.append(self.format(message))
        return "\n".join(formatted_messages)

    def format(self, message):
        """
        Format a message in vLLM format.

        Args:
            message: Message object to format

        Returns:
            str: Formatted message in vLLM format
        """
        role = message["role"]
        content = message["content"]

        # Handle different content types
        if isinstance(content, str):
            return f"{role}: {content}"
        else:
            # For multimodal content, extract text parts
            text_parts = []
            for item in content:
                if item["type"] == "text" and "text" in item:
                    text_parts.append(item["text"])
            return f"{role}: {' '.join(text_parts)}"


class OpenAIFormatter(Formatter):
    """
    Formatter for OpenAI format.
    """

    def format_data(self, data):
        """
        Format the data.

        Args:
            data: List of Conversation objects

        Returns:
            dict: Formatted data in OpenAI format
        """
        formatted_data = []
        for conversation in data:
            formatted_data.append(self.format_sample(conversation))
        return formatted_data

    def format_sample(self, sample):
        """
        Format a sample.

        Args:
            sample: Conversation object

        Returns:
            dict: Formatted sample in OpenAI format
        """
        formatted_messages = []
        for message in sample.messages:
            formatted_messages.append(self.format(message))
        return {"messages": formatted_messages}

    def format(self, message):
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
