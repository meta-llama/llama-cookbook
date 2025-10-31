import gc
import logging
from typing import Any, Dict, List

import torch
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

logger = logging.getLogger(__name__)


def create_inference_request(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_completion_tokens: int = 4096,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create an inference request for the model.

    Args:
        messages: List of message dictionaries for the conversation
        model: Model name to use for inference
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_completion_tokens: Maximum tokens to generate
        seed: Random seed for reproducibility
        use_json_decode: Whether to use JSON-guided decoding
        json_schema: JSON schema for guided decoding (required if use_json_decode=True)

    Returns:
        Dict containing the formatted request parameters
    """
    # strip assistant outputs
    messages = [m for m in messages if m["role"] != "assistant"]

    try:
        request = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_completion_tokens,
            "seed": seed,
            "messages": messages,
        }

        # Add JSON-guided decoding if requested
        # if use_json_decode:
        #     if json_schema is None:
        #         raise ValueError("json_schema is required when use_json_decode=True")
        #     request["response_format"] = {
        #         "type": "json_schema",
        #         "json_schema": {"name": "ExtractionSchema", "schema": json_schema},
        #     }

        return request

    except Exception as e:
        logger.error(f"Failed to create inference request: {e}")
        raise


class ModelRunner:
    """Base class for model inference runners."""

    def run_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Run inference on a batch of requests.

        Args:
            requests: List of request parameters

        Returns:
            List of raw text responses
        """
        # Abstract method, to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement run_batch")


# Not supported
class APIModelRunner(ModelRunner):
    """Runner for API-based model inference."""

    def __init__(self, api_key: str, base_url: str):
        """Initialize the API client."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def run_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Run inference on a batch of requests using the API.

        Args:
            requests: List of request parameters

        Returns:
            List of raw text responses
        """
        raise NotImplementedError(
            "Not supported at the moment. Use local model instead."
        )
        responses = []
        for request in tqdm(requests, desc="API Inference"):
            try:
                response = self.client.chat.completions.create(**request)
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        return responses


class LocalModelRunner(ModelRunner):
    """Runner for local model inference."""

    def __init__(
        self,
        ckpt_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        max_num_seqs: int = 128,
        enforce_eager: bool = True,
    ):
        """Initialize the local model."""
        try:
            self.model = LLM(
                ckpt_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager,
            )
            logger.info(f"Initialized local model: {ckpt_path}")

        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            raise

    def run_batch(self, request_batch: List[Dict[str, Any]]) -> List[str]:
        """
        Run inference on a batch of requests using the local model.

        Args:
            requests: List of request parameters

        Returns:
            List of raw text responses
        """
        try:
            # Extract messages
            messages = [req["messages"] for req in request_batch]

            # Prepare sampling parameters
            common_sampling_params = {
                "top_p": request_batch[0]["top_p"],
                "temperature": request_batch[0]["temperature"],
                "max_tokens": request_batch[0]["max_tokens"],
                "seed": request_batch[0]["seed"],
            }

            # Handle JSON-guided decoding if present
            if "response_format" in request_batch[0]:
                sampling_params = []
                for req in request_batch:
                    gd_params = GuidedDecodingParams(
                        json=req["response_format"]["json_schema"]["schema"]
                    )
                    sampling_params.append(
                        SamplingParams(
                            guided_decoding=gd_params, **common_sampling_params
                        )
                    )
            else:
                sampling_params = SamplingParams(**common_sampling_params)

            # Run inference
            outputs = self.model.chat(messages, sampling_params, use_tqdm=True)
            return [output.outputs[0].text for output in outputs]

        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            return [f"Error: {str(e)}" for _ in request_batch]

    def shutdown(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
