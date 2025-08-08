#!/usr/bin/env python
"""
Fine-tuning script for language models using torch tune.
Reads parameters from a config file and runs the torch tune command.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


## Will import from dataloader eventually
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


def run_torch_tune(config_path: str, args=None):
    """
    Run torch tune command with parameters from config file.

    Args:
        config_path: Path to the configuration file
        args: Command line arguments that may include additional kwargs to pass to the command
    """
    # Read the configuration
    config = read_config(config_path)

    # Extract parameters from config
    training_config = config.get("finetuning", {})

    # Initialize base_cmd to avoid "possibly unbound" error
    base_cmd = []

    # Determine the command based on configuration
    if training_config.get("distributed"):
        if training_config.get("strategy") == "lora":
            base_cmd = [
                "tune",
                "run",
                "--nproc_per_node",
                str(training_config.get("num_processes_per_node", 1)),
                "lora_finetune_distributed",
                "--config",
                training_config.get("torchtune_config"),
            ]
        elif training_config.get("strategy") == "fft":
            base_cmd = [
                "tune",
                "run",
                "--nproc_per_node",
                str(training_config.get("num_processes_per_node", 1)),
                "full_finetune_distributed",
                "--config",
                training_config.get("torchtune_config"),
            ]
        else:
            raise ValueError(f"Invalid strategy: {training_config.get('strategy')}")

    else:
        if training_config.get("strategy") == "lora":
            base_cmd = [
                "tune",
                "run",
                "lora_finetune_single_device",
                "--config",
                training_config.get("torchtune_config"),
            ]
        elif training_config.get("strategy") == "fft":
            base_cmd = [
                "tune",
                "run",
                "full_finetune_single_device",
                "--config",
                training_config.get("torchtune_config"),
            ]
        else:
            raise ValueError(f"Invalid strategy: {training_config.get('strategy')}")

    # Check if we have a valid command
    if not base_cmd:
        raise ValueError(
            "Could not determine the appropriate command based on the configuration"
        )

    # Add any additional kwargs if provided
    if args and args.kwargs:
        # Split the kwargs string by spaces to get individual key=value pairs
        kwargs_list = args.kwargs.split()
        base_cmd.extend(kwargs_list)
        logger.info(f"Added additional kwargs: {kwargs_list}")

    # Log the command
    logger.info(f"Running command: {' '.join(base_cmd)}")

    # Run the command
    try:
        subprocess.run(base_cmd, check=True)
        logger.info("Training complete!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)


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

    run_torch_tune(args.config, args=args)


if __name__ == "__main__":
    main()