class Message:
    """
    Data class representing a message with a role and content.
    """

    def __init__(self, role, content,content_type):
        self.role = role
        self.content = content
        self.type = content_type


class Formatter:
    """
    Base class for formatters that convert messages to different formats.
    """

    def __init__(self, message):
        self.message = message

    def format_data(self, data):
        """
        Format the message. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement format()")

    def format_sample(self, sample):
        """
        Format a sample. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement format_sample()")

    def read_data(self, data):
        """
        Format a sample. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement format_sample()")


class TorchtuneFormatter(Formatter):
    """
    Formatter for Torchtune format.
    """

    data = None

    def read_data(self, data):
        """
        Format a sample. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement format_sample()")

    def format(self):
        """
        Format the message in Torchtune format.
        """
        # Implementation for Torchtune format
        return {"role": self.message.role, "content": self.message.content}


class vLLMFormatter(Formatter):
    """
    Formatter for vLLM format.
    """

    def format(self):
        """
        Format the message in vLLM format.
        """
        # Implementation for vLLM format
        return f"{self.message.role}: {self.message.content}"


class OpenAIFormatter(Formatter):
    """
    Formatter for Hugging Face format.
    """

    def format(self):
        """
        Format the message in Hugging Face format.
        """
        # Implementation for OpenAI format
        raise NotImplementedError("Subclasses must implement format_sample()")
