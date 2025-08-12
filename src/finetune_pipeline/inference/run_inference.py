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
    save_formatted_data,
)

# Set up logging
logger = logging.getLogger(__name__)


def load_inference_data(
    inference_data_kwargs: Dict, formatter_config: Optional[Dict] = None
) -> List[Dict]:
    """
    Load and format inference data using inference_data_kwargs configuration.

    Args:
        inference_data_kwargs: Dictionary containing all inference data parameters
        formatter_config: Fallback formatter configuration for compatibility

    Returns:
        List of formatted data dictionaries ready for inference

    Raises:
        ValueError: If required parameters are missing or invalid
        FileNotFoundError: If local file path doesn't exist
        Exception: For data loading or formatting errors
    """
    # Extract parameters from inference_data_kwargs
    path = inference_data_kwargs.get("data_path")
    if not path:
        raise ValueError("data_path is required in inference_data_kwargs")

    format_data = inference_data_kwargs.get("format_data", True)
    formatter_type = inference_data_kwargs.get("formatter_type", "vllm")
    max_samples = inference_data_kwargs.get("max_samples")
    is_local = inference_data_kwargs.get("is_local", False)
    split = inference_data_kwargs.get("split", "validation")

    # Build dataset_kwargs
    dataset_kwargs = {"split": split}
    if "dataset_kwargs" in inference_data_kwargs:
        dataset_kwargs.update(inference_data_kwargs["dataset_kwargs"])

    # Use formatter_config for column mapping if provided
    column_mapping = {}
    if formatter_config:
        column_mapping = formatter_config.get("column_mapping", {})

    # Validate formatter type
    valid_formatters = ["vllm", "torchtune", "openai"]
    if formatter_type not in valid_formatters:
        raise ValueError(
            f"Invalid formatter_type '{formatter_type}'. Must be one of {valid_formatters}"
        )

    logger.info(f"Loading inference data from: {path}")
    logger.info(f"Format data: {format_data}, Formatter type: {formatter_type}")
    if max_samples:
        logger.info(f"Max samples: {max_samples}")

    formatted_data = []

    if format_data:
        # Validate column mapping
        if not column_mapping:
            logger.warning("No column mapping provided. Using default column names.")
            column_mapping = {"input": "input", "output": "output"}

        # Check if local file exists
        if is_local:
            import os

            if not os.path.exists(path):
                raise FileNotFoundError(f"Local file not found: {path}")
            logger.info(f"Loading data from local file: {path}")
        else:
            logger.info(f"Loading data from Hugging Face dataset: {path}")

        # Load the data with progress tracking
        try:
            logger.info("Loading raw data...")
            data = load_data(path, is_local, **dataset_kwargs)

            # Apply sample limit if specified
            if max_samples and hasattr(data, "__len__") and len(data) > max_samples:
                logger.info(
                    f"Limiting dataset to {max_samples} samples (original: {len(data)})"
                )
                if hasattr(data, "select"):
                    # For HuggingFace datasets
                    data = data.select(range(max_samples))
                else:
                    # For other iterable data
                    data = list(data)[:max_samples]

            data_size = len(data) if hasattr(data, "__len__") else "unknown"
            logger.info(f"Successfully loaded {data_size} samples")

        except Exception as e:
            logger.error(f"Failed to load data from {path}: {e}")
            logger.error(f"Dataset kwargs: {dataset_kwargs}")
            raise RuntimeError(f"Data loading failed: {str(e)}") from e

        # Convert to conversations with progress tracking
        try:
            logger.info("Converting data to conversation format...")
            conversations = convert_to_conversations(data, column_mapping)
            logger.info(f"Created {len(conversations)} conversations")

            # Validate conversations
            if not conversations:
                raise ValueError("No conversations were created from the data")

            # Log sample conversation for debugging
            if conversations and logger.isEnabledFor(logging.DEBUG):
                sample_conv = conversations[0]
                logger.debug(
                    f"Sample conversation: {sample_conv.messages[:2] if hasattr(sample_conv, 'messages') else sample_conv}"
                )

        except Exception as e:
            logger.error(f"Failed to convert data to conversations: {e}")
            logger.error(f"Column mapping: {column_mapping}")
            raise RuntimeError(f"Conversation conversion failed: {str(e)}") from e

        # Format conversations using specified formatter
        try:
            logger.info(f"Formatting conversations using {formatter_type} formatter...")
            formatter = get_formatter(formatter_type)

            # Add progress bar for large datasets
            if len(conversations) > 1000:
                logger.info("Processing large dataset with progress tracking...")
                from tqdm import tqdm

                formatted_data = []
                for conv in tqdm(conversations, desc="Formatting conversations"):
                    formatted_data.append(formatter.format_conversation(conv))
            else:
                formatted_data = formatter.format_data(conversations)

            logger.info(f"Successfully formatted {len(formatted_data)} samples")

            # Validate formatted data
            if not formatted_data:
                raise ValueError("No formatted data was produced")

            # Log sample formatted data for debugging
            if formatted_data and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample formatted data: {formatted_data[0]}")

        except Exception as e:
            logger.error(f"Failed to format conversations: {e}")
            logger.error(f"Formatter type: {formatter_type}")
            raise RuntimeError(f"Data formatting failed: {str(e)}") from e

    else:
        # Load pre-formatted data
        logger.info("Loading pre-formatted data...")
        try:
            import os
            from pathlib import Path

            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Pre-formatted file not found: {path}")

            # Support different file formats
            if file_path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    formatted_data = json.load(f)
            elif file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                # Support JSONL format
                formatted_data = []
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            if line.strip():  # Skip empty lines
                                formatted_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Skipping invalid JSON on line {line_num}: {e}"
                            )
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. Supported: .json, .jsonl, .ndjson"
                )

            # Apply sample limit if specified
            if max_samples and len(formatted_data) > max_samples:
                logger.info(
                    f"Limiting pre-formatted data to {max_samples} samples (original: {len(formatted_data)})"
                )
                formatted_data = formatted_data[:max_samples]

            logger.info(
                f"Successfully loaded {len(formatted_data)} pre-formatted samples"
            )

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load pre-formatted data from {path}: {e}")
            raise RuntimeError(f"Pre-formatted data loading failed: {str(e)}") from e

    # Final validation
    if not formatted_data:
        raise ValueError("No data was loaded. Check your path and configuration.")

    # Validate data structure
    if not isinstance(formatted_data, list):
        raise ValueError("Formatted data must be a list")

    # Basic structure validation for first sample
    if formatted_data:
        sample = formatted_data[0]
        if not isinstance(sample, dict):
            logger.warning(
                "First sample is not a dictionary. This might cause issues during inference."
            )

    logger.info(
        f"Data loading completed successfully. Total samples: {len(formatted_data)}"
    )
    return formatted_data


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
    inference_data: List[Dict],
    model_path: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 100,
    seed: Optional[int] = None,
    structured: bool = False,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 512,
    max_num_seqs: int = 1,
    tensor_parallel_size: int = 1,
) -> Dict[str, Any]:
    """
    Run inference on evaluation data using a vLLM server.

    Args:
        inference_data: Inference data to run inference on
        model_path: Path to the vLLM model
        temperature: Temperature for sampling
        top_p: Top-p for sampling
        max_tokens: Maximum number of tokens to generate
        seed: Random seed for reproducibility
        structured: Whether to use structured output
        gpu_memory_utilization: GPU memory utilization for the vLLM server
        max_model_len: Maximum model length for the vLLM server

    Returns:
        List of responses from the vLLM server
    """

    # Create an LLM
    logger.info(f"Initializing vLLM with model: {model_path}")
    try:
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
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
    logger.info(f"Running inference on {len(inference_data)} examples")
    outputs = vllm_call_batch(llm, inference_data, sampling_params)

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

    output_dir = config.get("output_dir", "/tmp/finetune-pipeline/")
    results_path = f"{output_dir}/data/inference_results.json"

    # Performance parameters
    gpu_memory_utilization = inference_config.get("gpu_memory_utilization", 0.95)
    max_model_len = inference_config.get("max_model_len", 512)
    max_num_seqs = inference_config.get("max_num_seqs", 1)
    tensor_parallel_size = inference_config.get("tensor_parallel_size", 1)
    dtype = inference_config.get("dtype", "auto")
    trust_remote_code = inference_config.get("trust_remote_code", False)

    # Generation parameters
    max_tokens = inference_config.get("max_tokens", 100)
    temperature = inference_config.get("temperature", 0.0)
    top_p = inference_config.get("top_p", 1.0)
    seed = inference_config.get("seed")
    structured = inference_config.get("structured", False)

    # Inference Data parameters
    inference_data_kwargs = inference_config.get("inference_data_kwargs", {})

    inference_data = load_inference_data(inference_data_kwargs, formatter_config)

    results = run_vllm_batch_inference_on_dataset(
        inference_data,
        model_path,
        temperature,
        top_p,
        max_tokens,
        seed,
        structured,
        gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        tensor_parallel_size,
    )

    save_inference_results(results, results_path)


if __name__ == "__main__":
    main()
