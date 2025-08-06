#!/usr/bin/env python
"""
Script to start a vLLM server for inference.

This script provides a convenient way to start a vLLM server with various configuration options.
It supports loading models from local paths or Hugging Face model IDs.

Example usage:
    python start_vllm_server.py --model-path meta-llama/Llama-2-7b-chat-hf
    python start_vllm_server.py --model-path /path/to/local/model --port 8080
    python start_vllm_server.py --config /path/to/config.yaml
    python start_vllm_server.py  # Uses the default config.yaml in the parent directory
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Try to import yaml for config file support
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed. Config file support limited to JSON format.")


def read_config(config_path: str) -> Dict:
    """
    Read the configuration file (supports both JSON and YAML formats).

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration parameters

    Raises:
        ValueError: If the file format is not supported
        ImportError: If the required package for the file format is not installed
    """
    file_extension = Path(config_path).suffix.lower()

    with open(config_path, "r") as f:
        if file_extension in [".json"]:
            config = json.load(f)
        elif file_extension in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise ImportError(
                    "The 'pyyaml' package is required to load YAML files. "
                    "Please install it with 'pip install pyyaml'."
                )
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {file_extension}. "
                f"Supported formats are: .json, .yaml, .yml"
            )

    return config


def check_vllm_installed() -> bool:
    """
    Check if vLLM is installed.

    Returns:
        bool: True if vLLM is installed, False otherwise
    """
    try:
        subprocess.run(
            ["vllm", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return True
    except FileNotFoundError:
        return False


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    quantization: Optional[str] = None,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    additional_args: Optional[Dict] = None,
) -> None:
    """
    Start a vLLM server with the specified parameters.

    Args:
        model_path: Path to the model or Hugging Face model ID
        port: Port to run the server on
        host: Host to run the server on
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        max_model_len: Maximum sequence length
        max_num_seqs: Maximum number of sequences
        quantization: Quantization method (e.g., "awq", "gptq", "squeezellm")
        dtype: Data type for model weights (e.g., "half", "float", "bfloat16", "auto")
        gpu_memory_utilization: Fraction of GPU memory to use
        trust_remote_code: Whether to trust remote code when loading the model
        enforce_eager: Whether to enforce eager execution
        additional_args: Additional arguments to pass to vLLM

    Raises:
        subprocess.CalledProcessError: If the vLLM server fails to start
        FileNotFoundError: If vLLM is not installed
    """
    # Check if vLLM is installed
    if not check_vllm_installed():
        logger.error(
            "vLLM is not installed. Please install it with 'pip install vllm'."
        )
        sys.exit(1)

    # Build the command
    cmd = ["vllm", "serve", model_path]

    # Add basic parameters
    cmd.extend(["--port", str(port)])
    cmd.extend(["--host", host])
    cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    cmd.extend(["--max-model-len", str(max_model_len)])
    cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    cmd.extend(["--dtype", dtype])

    # Add optional parameters
    if quantization:
        cmd.extend(["--quantization", quantization])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if enforce_eager:
        cmd.append("--enforce-eager")

    # Add additional arguments
    if additional_args:
        for key, value in additional_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    # Log the command
    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("vLLM server stopped by user.")
        sys.exit(0)


def find_config_file():
    """
    Find the config.yaml file in the parent directory.

    Returns:
        str: Path to the config file
    """
    # Try to find the config file in the parent directory
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    config_path = parent_dir / "config.yaml"

    if config_path.exists():
        return str(config_path)
    else:
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start a vLLM server for inference")

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file (JSON or YAML)",
    )

    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Path to the model or Hugging Face model ID",
    )
    model_group.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "squeezellm"],
        help="Quantization method to use",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["half", "float", "bfloat16", "auto"],
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model",
    )

    # Server options
    server_group = parser.add_argument_group("Server")
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )

    # Performance options
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    perf_group.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    perf_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences",
    )
    perf_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use",
    )
    perf_group.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution",
    )

    args = parser.parse_args()

    # Load config file
    config = {}
    config_path = args.config

    # If no config file is provided, try to find the default one
    if not config_path:
        config_path = find_config_file()
        if config_path:
            logger.info(f"Using default config file: {config_path}")

    if config_path:
        try:
            config = read_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            sys.exit(1)

    # Extract inference section from config if it exists
    inference_config = config.get("inference", {})

    # Merge command-line arguments with config file
    # Command-line arguments take precedence
    model_path = args.model_path or inference_config.get("model_path")
    if not model_path:
        logger.error(
            "Model path must be provided either via --model-path or in the config file under inference.model_path"
        )
        sys.exit(1)

    # Extract parameters
    params = {
        "model_path": model_path,
        "port": (
            args.port
            if args.port != parser.get_default("port")
            else inference_config.get("port", args.port)
        ),
        "host": (
            args.host
            if args.host != parser.get_default("host")
            else inference_config.get("host", args.host)
        ),
        "tensor_parallel_size": (
            args.tensor_parallel_size
            if args.tensor_parallel_size != parser.get_default("tensor_parallel_size")
            else inference_config.get("tensor_parallel_size", args.tensor_parallel_size)
        ),
        "max_model_len": (
            args.max_model_len
            if args.max_model_len != parser.get_default("max_model_len")
            else inference_config.get("max_model_len", args.max_model_len)
        ),
        "max_num_seqs": (
            args.max_num_seqs
            if args.max_num_seqs != parser.get_default("max_num_seqs")
            else inference_config.get("max_num_seqs", args.max_num_seqs)
        ),
        "quantization": args.quantization or inference_config.get("quantization"),
        "dtype": (
            args.dtype
            if args.dtype != parser.get_default("dtype")
            else inference_config.get("dtype", args.dtype)
        ),
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
            if args.gpu_memory_utilization
            != parser.get_default("gpu_memory_utilization")
            else inference_config.get(
                "gpu_memory_utilization", args.gpu_memory_utilization
            )
        ),
        "trust_remote_code": args.trust_remote_code
        or inference_config.get("trust_remote_code", False),
        "enforce_eager": args.enforce_eager
        or inference_config.get("enforce_eager", False),
    }

    # Get additional arguments from inference config
    additional_args = {k: v for k, v in inference_config.items() if k not in params}
    if additional_args:
        params["additional_args"] = additional_args

    # Start the vLLM server
    start_vllm_server(**params)


if __name__ == "__main__":
    main()