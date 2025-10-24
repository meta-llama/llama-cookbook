"""Groq API client wrapper for PPTX to Transcript."""

from typing import Optional, Any, Union
import os
from .image_processing import encode_image
from ..config.settings import get_api_config, get_system_prompt, is_knowledge_enabled, get_knowledge_config

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class GroqClient:
    """Wrapper for Groq API client with configuration management."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Groq client.

        Args:
            api_key: API key for Groq. If None, will be loaded from environment.
            model: Model to use. If None, uses default vision model.
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq package is required. Install with: pip install groq")

        if api_key is None:
            api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable or provide api_key parameter.")

        self.client = Groq(api_key=api_key)

        # Groq vision models (update as new models become available)
        self.model = model or "meta-llama/llama-4-maverick-17b-128e-instruct"

        # Groq API configuration
        self.max_tokens = 4096
        self.temperature = 0.1

    def generate_transcript(
        self,
        image_path: str,
        speaker_notes: str = "",
        system_prompt: Optional[str] = None,
        context_bundle: Optional[Any] = None,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Generate transcript from slide image and speaker notes with optional context.

        Args:
            image_path: Path to the slide image
            speaker_notes: Speaker notes for the slide
            system_prompt: Custom system prompt. If None, uses default from config.
            context_bundle: ContextBundle for knowledge integration
            stream: Whether to stream the response

        Returns:
            Generated transcript text if not streaming, otherwise the response object
        """
        if system_prompt is None:
            system_prompt = get_system_prompt()

        # Enhance with context if available
        if context_bundle is not None and is_knowledge_enabled():
            system_prompt, user_message_prefix = self._integrate_context(
                system_prompt, context_bundle
            )
        else:
            user_message_prefix = ""

        encoded_image = encode_image(image_path)

        # Build user message with optional context prefix
        user_text = f"{user_message_prefix}Speaker Notes: {speaker_notes}".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text,
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream,
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    def _integrate_context(self, system_prompt: str, context_bundle: Any) -> tuple[str, str]:
        """
        Integrate context bundle into system prompt or user message.

        Args:
            system_prompt: Original system prompt
            context_bundle: ContextBundle with context information

        Returns:
            Tuple of (enhanced_system_prompt, user_message_prefix)
        """
        try:
            # Import here to avoid circular imports
            from ..knowledge.context_manager import ContextManager

            context_manager = ContextManager()
            knowledge_config = get_knowledge_config()
            integration_method = knowledge_config.get('context', {}).get('integration_method', 'system_prompt')

            # Get formatted context
            context_data = context_manager.get_context_for_integration(
                context_bundle, integration_method
            )

            if not context_data:
                return system_prompt, ""

            if integration_method == "system_prompt":
                return self._enhance_system_prompt(system_prompt, context_data), ""
            elif integration_method == "user_message":
                return system_prompt, self._enhance_user_message(context_data)
            else:
                return system_prompt, ""

        except Exception as e:
            # Graceful degradation - log error but continue without context
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to integrate context: {e}")
            return system_prompt, ""

    def _enhance_system_prompt(self, system_prompt: str, context_data: dict) -> str:
        """
        Enhance system prompt with context information.

        Args:
            system_prompt: Original system prompt
            context_data: Context data from ContextManager

        Returns:
            Enhanced system prompt
        """
        context_addition = context_data.get('context_addition', '')
        integration_point = context_data.get('integration_point', 'before_instructions')

        if not context_addition:
            return system_prompt

        if integration_point == 'before_instructions':
            # Add context before the main instructions
            enhanced_prompt = f"{context_addition}\n\n{system_prompt}"
        else:
            # Default: append at the end
            enhanced_prompt = f"{system_prompt}\n\n{context_addition}"

        return enhanced_prompt

    def _enhance_user_message(self, context_data: dict) -> str:
        """
        Create user message prefix with context information.

        Args:
            context_data: Context data from ContextManager

        Returns:
            User message prefix
        """
        context_addition = context_data.get('context_addition', '')

        if context_addition:
            return f"{context_addition}\n\n"

        return ""

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

        try:
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream,
            )

            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
            else:
                return response

        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    def list_models(self) -> list:
        """
        List available models from Groq.

        Returns:
            List of available models
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def get_model_info(self) -> dict:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'provider': 'Groq'
        }
