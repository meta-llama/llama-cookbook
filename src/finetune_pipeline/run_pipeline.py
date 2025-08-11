#!/usr/bin/env python
"""
End-to-end pipeline for data loading, fine-tuning, and inference.

This script integrates all the modules in the finetune_pipeline package:
1. Data loading and formatting
2. Model fine-tuning
3. vLLM server startup
4. Inference on the fine-tuned model

Example usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --skip-finetuning --skip-server
    python run_pipeline.py --config config.yaml --only-inference
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Import modules from the finetune_pipeline package
from finetune_pipeline.data.data_loader import load_and_format_data, read_config
from finetune_pipeline.finetuning.run_finetuning import run_torch_tune

from finetune_pipeline.inference.run_inference import (
    run_vllm_batch_inference_on_dataset,
)
from finetune_pipeline.inference.save_inference_results import save_inference_results
from finetune_pipeline.inference.start_vllm_server import start_vllm_server


def run_data_loading(config_path: str) -> Tuple[List[str], List[str]]:
    """
    Run the data loading and formatting step.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple containing lists of paths to the formatted data and conversation data
    """
    logger.info("=== Step 1: Data Loading and Formatting ===")

    # Read the configuration
    config = read_config(config_path)
    formatter_config = config.get("formatter", {})
    output_dir = config.get("output_dir", "/tmp/finetune-pipeline/data/")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and format the data
    try:
        formatted_data_paths, conversation_data_paths = load_and_format_data(
            formatter_config, output_dir
        )
        logger.info(f"Data loading and formatting complete. Saved to {output_dir}")
        logger.info(f"Formatted data paths: {formatted_data_paths}")
        logger.info(f"Conversation data paths: {conversation_data_paths}")
        return formatted_data_paths, conversation_data_paths
    except Exception as e:
        logger.error(f"Error during data loading and formatting: {e}")
        raise


def run_finetuning(config_path: str, formatted_data_paths: List[str]) -> str:
    """
    Run the fine-tuning step.

    Args:
        config_path: Path to the configuration file
        formatted_data_paths: Paths to the formatted data

    Returns:
        Path to the fine-tuned model
    """
    logger.info("=== Step 2: Model Fine-tuning ===")

    # Read the configuration
    config = read_config(config_path)
    finetuning_config = config.get("finetuning", {})

    # Get the path to the formatted data for the train split
    train_data_path = None
    for path in formatted_data_paths:
        if "train_" in path:
            train_data_path = path
            break

    if not train_data_path:
        logger.warning("No train split found in formatted data. Using the first file.")
        train_data_path = formatted_data_paths[0]

    # Prepare additional kwargs for the fine-tuning
    kwargs = f"dataset=finetune_pipeline.finetuning.custom_sft_dataset dataset.train_on_input=True dataset.dataset_path={train_data_path}"

    # Create an args object to pass to run_torch_tune
    class Args:
        pass

    args = Args()
    args.kwargs = kwargs

    # Run the fine-tuning
    try:
        logger.info(f"Starting fine-tuning with data from {train_data_path}")
        run_torch_tune(finetuning_config, args=args)

        # Get the path to the latest chekpoint of the fine-tuned model
        model_output_dir = finetuning_config.get("output_dir", config.get("output_dir"))
        epochs = finetuning_config.get("epochs", 1)
        checkpoint_path = os.path.join(model_output_dir, f"epochs_{epochs-1}")
        logger.info(
            f"Fine-tuning complete. Latest checkpoint saved to {checkpoint_path}"
        )
        return checkpoint_path
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


def run_vllm_server(config_path: str, model_path: str) -> str:
    """
    Start the vLLM server.

    Args:
        config_path: Path to the configuration file
        model_path: Path to the fine-tuned model

    Returns:
        URL of the vLLM server
    """
    logger.info("=== Step 3: Starting vLLM Server ===")

    # Read the configuration
    config = read_config(config_path)
    inference_config = config.get("inference", {})

    model_path = inference_config.get(
        "model_path", "/home/ubuntu/yash-workspace/medgemma-4b-it"
    )

    # # Update the model path in the inference config
    # inference_config["model_path"] = model_path

    # Extract server parameters
    port = inference_config.get("port", 8000)
    host = inference_config.get("host", "0.0.0.0")
    tensor_parallel_size = inference_config.get("tensor_parallel_size", 1)
    max_model_len = inference_config.get("max_model_len", 4096)
    max_num_seqs = inference_config.get("max_num_seqs", 256)
    quantization = inference_config.get("quantization")
    gpu_memory_utilization = inference_config.get("gpu_memory_utilization", 0.9)
    enforce_eager = inference_config.get("enforce_eager", False)

    # Start the server in a separate process
    try:
        logger.info(f"Starting vLLM server with model {model_path}")
        result = start_vllm_server(
            model_path,
            port,
            host,
            tensor_parallel_size,
            max_model_len,
            max_num_seqs,
            quantization,
            gpu_memory_utilization,
            enforce_eager,
        )
        if result.returncode == 0:
            server_url = f"http://{host}:{port}/v1"
            logger.info(f"vLLM server started at {server_url}")
            return server_url
        else:
            logger.error(f"vLLM server failed to start")
            raise RuntimeError("vLLM server failed to start")
    except Exception as e:
        logger.error(f"Error starting vLLM server: {e}")
        raise


def run_inference(
    config_path: str, formatted_data_paths: List[str], model_path: str = ""
) -> str:
    """
    Run inference on the fine-tuned model.

    Args:
        config_path: Path to the configuration file
        server_url: URL of the vLLM server
        formatted_data_paths: Paths to the formatted data

    Returns:
        Path to the inference results
    """
    logger.info("=== Step 4: Running Inference ===")

    config = read_config(config_path)
    inference_config = config.get("inference", {})
    formatter_config = config.get("formatter", {})
    output_dir = config.get("output_dir", "/tmp/finetune-pipeline/")

    # Model parameters
    if model_path == "":
        model_path = inference_config.get("model_path", None)
        if model_path is None:
            raise ValueError("model_path must be specified in the config")

    # Get data path from parameters or config
    inference_data_path = inference_config.get("inference_data", None)
    if inference_data_path is None:
        raise ValueError("Inference data path must be specified in config")
    output_path = f"{output_dir}/inference_results.json"

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

    # Run inference
    try:
        logger.info(f"Running inference on {inference_data_path}")
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

        # Save the results
        results_path = os.path.join(output_dir, "inference_results.json")
        save_inference_results(results, results_path)

        logger.info(f"Inference complete. Results saved to {results_path}")
        return results_path
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


def run_pipeline(
    config_path: str,
    skip_data_loading: bool = False,
    skip_finetuning: bool = False,
    skip_server: bool = False,
    skip_inference: bool = False,
    only_data_loading: bool = False,
    only_finetuning: bool = False,
    only_server: bool = False,
    only_inference: bool = False,
) -> None:
    """
    Run the end-to-end pipeline.

    Args:
        config_path: Path to the configuration file
        skip_data_loading: Whether to skip the data loading step
        skip_finetuning: Whether to skip the fine-tuning step
        skip_server: Whether to skip starting the vLLM server
        skip_inference: Whether to skip the inference step
        only_data_loading: Whether to run only the data loading step
        only_finetuning: Whether to run only the fine-tuning step
        only_server: Whether to run only the vLLM server step
        only_inference: Whether to run only the inference step
    """
    logger.info(f"Starting pipeline with config {config_path}")

    # Check if the config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist")
        sys.exit(1)

    # Handle "only" flags
    if only_data_loading:
        skip_finetuning = True
        skip_server = True
        skip_inference = True
    elif only_finetuning:
        skip_data_loading = True
        skip_server = True
        skip_inference = True
    elif only_server:
        skip_data_loading = True
        skip_finetuning = True
        skip_inference = True
    elif only_inference:
        skip_data_loading = True
        skip_finetuning = True
        skip_server = True

    # Step 1: Data Loading and Formatting
    formatted_data_paths = []
    conversation_data_paths = []
    if not skip_data_loading:
        try:
            formatted_data_paths, conversation_data_paths = run_data_loading(
                config_path
            )
        except Exception as e:
            logger.error(f"Pipeline failed at data loading step: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping data loading step")
        # Try to infer the paths from the config
        config = read_config(config_path)
        output_dir = config.get("output_dir", "/tmp/finetune-pipeline/data/")
        formatter_type = config.get("formatter", {}).get("type", "torchtune")
        # This is a simplification; in reality, you'd need to know the exact paths
        formatted_data_paths = [
            os.path.join(output_dir, f"train_{formatter_type}_formatted_data.json")
        ]

    # Step 2: Fine-tuning
    model_path = ""
    if not skip_finetuning:
        try:
            model_path = run_finetuning(config_path, formatted_data_paths)
        except Exception as e:
            logger.error(f"Pipeline failed at fine-tuning step: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping fine-tuning step")
        # Try to infer the model path from the config
        config = read_config(config_path)
        output_dir = config.get("output_dir", "/tmp/finetune-pipeline/")
        model_path = os.path.join(output_dir, "finetuned_model")

    # # Step 3: Start vLLM Server
    # server_url = ""
    # server_process = None
    # if not skip_server:
    #     try:
    #         server_url = run_vllm_server(config_path, model_path)
    #     except Exception as e:
    #         logger.error(f"Pipeline failed at vLLM server step: {e}")
    #         sys.exit(1)
    # else:
    #     logger.info("Skipping vLLM server step")
    #     # Try to infer the server URL from the config
    #     config = read_config(config_path)
    #     inference_config = config.get("inference", {})
    #     host = inference_config.get("host", "0.0.0.0")
    #     port = inference_config.get("port", 8000)
    #     server_url = f"http://{host}:{port}/v1"

    # Step 3: Inference
    if not skip_inference:
        try:
            results_path = run_inference(config_path, formatted_data_paths, model_path)
            logger.info(
                f"Pipeline completed successfully. Results saved to {results_path}"
            )
        except Exception as e:
            logger.error(f"Pipeline failed at inference step: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping inference step")

    logger.info("Pipeline execution complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )

    # Skip flags
    parser.add_argument(
        "--skip-data-loading",
        action="store_true",
        help="Skip the data loading step",
    )
    parser.add_argument(
        "--skip-finetuning",
        action="store_true",
        help="Skip the fine-tuning step",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip starting the vLLM server",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference step",
    )

    # Only flags
    parser.add_argument(
        "--only-data-loading",
        action="store_true",
        help="Run only the data loading step",
    )
    parser.add_argument(
        "--only-finetuning",
        action="store_true",
        help="Run only the fine-tuning step",
    )
    parser.add_argument(
        "--only-server",
        action="store_true",
        help="Run only the vLLM server step",
    )
    parser.add_argument(
        "--only-inference",
        action="store_true",
        help="Run only the inference step",
    )

    args = parser.parse_args()

    # Run the pipeline
    run_pipeline(
        config_path=args.config,
        skip_data_loading=args.skip_data_loading,
        skip_finetuning=args.skip_finetuning,
        skip_server=args.skip_server,
        skip_inference=args.skip_inference,
        only_data_loading=args.only_data_loading,
        only_finetuning=args.only_finetuning,
        only_server=args.only_server,
        only_inference=args.only_inference,
    )


if __name__ == "__main__":
    main()
