import json
import logging
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from tqdm import tqdm

from vllm import SamplingParams

from ..data.data_loader import convert_to_conversations, get_formatter, load_data


class VLLMInferenceRequest(TypedDict):
    """Type definition for VLLM inference request format."""

    messages: List[List[Dict[str, Any]]]
    sampling_params: Union[SamplingParams, List[SamplingParams]]


class VLLMClient:
    """Client for interacting with a vLLM server."""

    def __init__(self, server_url: str = "http://localhost:8000/v1"):
        """
        Initialize the vLLM client.

        Args:
            server_url: URL of the vLLM server
        """
        self.server_url = server_url
        self.logger = logging.getLogger(__name__)

    def generate(self, request: VLLMInferenceRequest) -> Dict[str, Any]:
        """
        Send a request to the vLLM server and get the response.

        Args:
            request: The inference request

        Returns:
            The response from the vLLM server

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        # Format the request for the OpenAI-compatible API
        vllm_request = {
            "messages": request.get("messages", []),
            "temperature": request.get("temperature", 0.7),
            "top_p": request.get("top_p", 1.0),
            "max_tokens": request.get("max_completion_tokens", 100),
        }

        if "seed" in request:
            vllm_request["seed"] = request["seed"]

        if "response_format" in request:
            vllm_request["response_format"] = request["response_format"]

        # Send the request to the vLLM server
        try:
            response = requests.post(
                f"{self.server_url}/chat/completions",
                json=vllm_request,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending request to vLLM server: {e}")
            raise