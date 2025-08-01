"""Llama API client wrapper for PPTX to Transcript."""

from typing import Optional, Any, Union
from llama_api_client import LlamaAPIClient
from .image_processing import encode_image
from ..config.settings import get_api_config, get_system_prompt


class LlamaClient:
    """Wrapper for Llama API client with configuration management."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Llama client.

        Args:
            api_key: API key for Llama. If None, will be loaded from config/environment.
        """
        api_config = get_api_config()

        if api_key is None:
            api_key = api_config.get('llama_api_key')

        if not api_key:
            raise ValueError("Llama API key not found. Set LLAMA_API_KEY environment variable or provide api_key parameter.")

        self.client = LlamaAPIClient(api_key=api_key)
        self.model = api_config.get('llama_model', 'Llama-4-Maverick-17B-128E-Instruct-FP8')

    def generate_transcript(
        self,
        image_path: str,
        speaker_notes: str = "",
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Generate transcript from slide image and speaker notes.

        Args:
            image_path: Path to the slide image
            speaker_notes: Speaker notes for the slide
            system_prompt: Custom system prompt. If None, uses default from config.
            stream: Whether to stream the response

        Returns:
            Generated transcript text if not streaming, otherwise the response object
        """
        if system_prompt is None:
            system_prompt = get_system_prompt()

        encoded_image = encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Speaker Notes: {speaker_notes}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}",
                            },
                        },
                    ],
                },
            ],
            stream=stream,
        )

        if stream:
            return response
        else:
            return response.completion_message.content.text

    def run(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Legacy method for backward compatibility with notebook code.

        Args:
            image_path: Path to the image file
            system_prompt: System prompt for the chat completion
            user_prompt: User prompt (speaker notes)
            stream: Whether to stream the response

        Returns:
            Response from the chat completion
        """
        encoded_image = encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Speaker Notes: {user_prompt}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}",
                            },
                        },
                    ],
                },
            ],
            stream=stream,
        )

        if stream:
            for chunk in response:
                print(chunk.event.delta.text, end="", flush=True)
        else:
            return response
