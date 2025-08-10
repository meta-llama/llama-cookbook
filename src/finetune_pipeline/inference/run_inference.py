import argparse
import json
import logging
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from tqdm import tqdm

from vllm import LLM, SamplingParams

from ..data.data_loader import (
    convert_to_conversations,
    get_formatter,
    load_data,
    read_config,
)

# Set up logging
logger = logging.getLogger(__name__)


def vllm_call_batch(
    llm: LLM, data: List[Dict], sampling_params: SamplingParams
) -> List[str]:
    """
    Process a batch of data through vLLM.

    Args:
        llm: vLLM model instance
        data: List of formatted data for inference
        sampling_params: Parameters for text generation
        batch_size: Number of items to process in each batch

    Returns:
        List of generated text responses
    """
    messages_batch = []
    for d in data:
        messages_batch.append(d["messages"])

    try:
        responses = llm.chat(messages_batch, sampling_params, use_tqdm=True)

        outputs = []
        for response in responses:
            text = ""
            for output in response.outputs:
                text += output.text
            outputs.append(text)
        return outputs
    except Exception as e:
        logger.error(f"Error during vLLM inference: {e}")
        raise


def run_vllm_batch_inference_on_dataset(
    inference_data_path: str,
    model_path: str,
    is_local: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 100,
    seed: Optional[int] = None,
    structured: bool = False,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 4096,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run inference on evaluation data using a vLLM server.

    Args:
        eval_data_path: Path to the evaluation data
        server_url: URL of the vLLM server
        is_local: Whether the data is stored locally
        temperature: Temperature for sampling
        top_p: Top-p for sampling
        max_tokens: Maximum number of tokens to generate
        seed: Random seed for reproducibility
        dataset_kwargs: Additional arguments to pass to the load_dataset function
        column_mapping: Mapping of column names

    Returns:
        List of responses from the vLLM server
    """

    logger.info(f"Loading Inference data from {inference_data_path}")

    # Load the evaluation data
    if dataset_kwargs is None:
        dataset_kwargs = {}

    try:
        data = load_data(inference_data_path, is_local, **dataset_kwargs)
    except Exception as e:
        logger.error(f"Failed to load data from {inference_data_path}: {e}")
        raise

    # Convert the data to conversations
    logger.info("Converting data to conversation format")
    conversations = convert_to_conversations(data, column_mapping)

    # Convert the conversations to vLLM format
    vllm_formatter = get_formatter("vllm")
    formatted_data = vllm_formatter.format_data(conversations)

    # Create an LLM
    logger.info(f"Initializing vLLM with model: {model_path}")
    try:
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM model: {e}")
        raise

    # Configure sampling parameters
    if structured and seed is not None:
        # If structured output is needed, guided_decoding can be configured here
        # For now, we're not using guided_decoding but this allows for future expansion
        guided_decoding_params = None
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            guided_decoding=guided_decoding_params,
            seed=seed,
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

    # Run inference on the formatted data
    logger.info(f"Running inference on {len(formatted_data)} examples")
    outputs = vllm_call_batch(llm, formatted_data, sampling_params)

    # Return a dictionary containing the outputs and metadata
    return {
        "outputs": outputs,
        "model_path": model_path,
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    }


def save_inference_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save inference results to a file.

    Args:
        results: Dictionary containing inference results
        output_path: Path to save the results
    """
    import json
    from pathlib import Path

    # Create directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract only serializable data
    serializable_results = {
        "model_path": results["model_path"],
        "params": results["params"],
        "outputs": results["outputs"],
    }

    logger.info(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Saved inference results to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model using torch tune"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON or YAML)",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        default=None,
        help="Additional key-value pairs to pass to the command (space-separated, e.g., 'dataset=module.function dataset.param=value')",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    inference_config = config.get("inference", {})
    formatter_config = config.get("formatter", {})

    # Model parameters
    model_path = inference_config.get("model_path", None)
    if model_path is None:
        raise ValueError("model_path must be specified in the config")

    # Get data path from parameters or config
    inference_data_path = inference_config.get("inference_data", None)
    if inference_data_path is None:
        raise ValueError("Inference data path must be specified in config")
    output_dir = config.get("output_dir", "/tmp/finetune-pipeline/")
    results_path = f"{output_dir}/inference_results.json"

    # Performance parameters
    gpu_memory_utilization = inference_config.get("gpu_memory_utilization", 0.95)
    max_model_len = inference_config.get("max_model_len", 512)
    tensor_parallel_size = inference_config.get("tensor_parallel_size", 1)
    dtype = inference_config.get("dtype", "auto")
    trust_remote_code = inference_config.get("trust_remote_code", False)

    # Generation parameters
    max_tokens = inference_config.get("max_tokens", 100)
    temperature = inference_config.get("temperature", 0.0)
    top_p = inference_config.get("top_p", 1.0)
    seed = inference_config.get("seed")
    structured = inference_config.get("structured", False)

    # Data parameters
    is_local = formatter_config.get("is_local", False)
    dataset_kwargs = formatter_config.get("dataset_kwargs", {})
    column_mapping = formatter_config.get("column_mapping", {})

    results = run_vllm_batch_inference_on_dataset(
        inference_data_path,
        model_path,
        is_local,
        temperature,
        top_p,
        max_tokens,
        seed,
        structured,
        gpu_memory_utilization,
        max_model_len,
        dataset_kwargs,
        column_mapping,
    )

    save_inference_results(results, results_path)


if __name__ == "__main__":
    main()
