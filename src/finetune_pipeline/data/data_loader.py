"""
Data loader module for loading and formatting data from Hugging Face.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Try to import yaml, but don't fail if it's not available
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Try to import datasets, but don't fail if it's not available
try:
    from datasets import load_dataset, load_from_disk

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

    # Define dummy functions to avoid "possibly unbound" errors
    def load_dataset(*args, **kwargs):
        raise ImportError("The 'datasets' package is required to load data.")

    def load_from_disk(*args, **kwargs):
        raise ImportError("The 'datasets' package is required to load data.")


from .formatter import Formatter, OpenAIFormatter, TorchtuneFormatter, vLLMFormatter


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
            # Only use yaml if it's available (HAS_YAML is True here)
            import yaml  # This import will succeed because we've already checked HAS_YAML

            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {file_extension}. "
                f"Supported formats are: .json, .yaml, .yml"
            )

    return config


def load_data(data_path: str, is_local: bool = False, **kwargs):
    """
    Load data from Hugging Face Hub or local disk.

    Args:
        data_path: Path to the dataset (either a Hugging Face dataset ID or a local path)
        is_local: Whether the data is stored locally
        **kwargs: Additional arguments to pass to the load_dataset function

    Returns:
        Dataset object from the datasets library

    Raises:
        ImportError: If the datasets package is not installed
        ValueError: If data_path is None or empty
    """
    if not HAS_DATASETS:
        raise ImportError(
            "The 'datasets' package is required to load data. "
            "Please install it with 'pip install datasets'."
        )

    if not data_path:
        raise ValueError("data_path must be provided")

    if is_local:
        # Load from local disk
        dataset = load_from_disk(data_path)
    else:
        # Load from Hugging Face Hub
        dataset = load_dataset(data_path, **kwargs)

    return dataset


def get_formatter(formatter_type: str) -> Formatter:
    """
    Get the appropriate formatter based on the formatter type.

    Args:
        formatter_type: Type of formatter to use ('torchtune', 'vllm', or 'openai')

    Returns:
        Formatter: Formatter instance

    Raises:
        ValueError: If the formatter type is not supported
    """
    formatter_map = {
        "torchtune": TorchtuneFormatter,
        "vllm": vLLMFormatter,
        "openai": OpenAIFormatter,
    }

    if formatter_type.lower() not in formatter_map:
        raise ValueError(
            f"Unsupported formatter type: {formatter_type}. "
            f"Supported types are: {', '.join(formatter_map.keys())}"
        )

    return formatter_map[formatter_type.lower()]()


def convert_to_conversations(data, column_mapping: Optional[Dict] = None):
    """
    Convert data to a list of Conversation objects.

    Args:
        data: Data to convert
        column_mapping: Optional mapping of column names

    Returns:
        list: List of Conversation objects
    """
    # Import here to avoid circular imports
    from .formatter import Conversation

    # Default column mapping if none provided
    if column_mapping is None:
        column_mapping = {"input": "input", "output": "output", "image": "image"}

    # Validate column mapping
    required_fields = ["input", "output"]
    for field in required_fields:
        if field not in column_mapping:
            raise ValueError(f"Column mapping must include '{field}' field")

    conversations = []
    for item in data:
        # Extract fields from the dataset item using the column mapping
        image_field = column_mapping.get("image")
        input_field = column_mapping.get("input")
        output_field = column_mapping.get("output")

        image = item.get(image_field, None) if image_field else None
        input_text = item.get(input_field, "")
        output_label = item.get(output_field, "")

        # Create a new conversation
        conversation = Conversation()

        # Create user content and user message
        user_content = [
            {"type": "text", "text": input_text},
        ]
        # Add image to user content
        if image is not None:
            user_content.append({"type": "image", "image_url": {"url": image}})

        user_message = {"role": "user", "content": user_content}

        # Create assistant message with text content
        assistant_content = [
            {"type": "text", "text": output_label},
        ]
        assistant_message = {"role": "assistant", "content": assistant_content}

        # Add messages to the conversation
        conversation.add_message(user_message)
        conversation.add_message(assistant_message)

        # Add the conversation to the list
        conversations.append(conversation)

    return conversations


def format_data(data, formatter_type: str, column_mapping: Optional[Dict] = None):
    """
    Format the data using the specified formatter.

    Args:
        data: Data to format
        formatter_type: Type of formatter to use ('torchtune', 'vllm', or 'openai')
        column_mapping: Optional mapping of column names

    Returns:
        Formatted data in the specified format
    """
    # First convert the data to conversations
    conversations = convert_to_conversations(data, column_mapping)

    # Then get the formatter and format the conversations
    formatter = get_formatter(formatter_type)
    formatted_data = formatter.format_data(conversations)

    return formatted_data


def load_and_format_data(config_path: str):
    """
    Load and format data based on the configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Formatted data in the specified format
    """
    # Read the configuration
    config = read_config(config_path)

    # Extract parameters from config
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("data_path must be specified in the config file")

    is_local = config.get("is_local", False)
    formatter_type = config.get("formatter_type", "torchtune")
    column_mapping = config.get("column_mapping")
    dataset_kwargs = config.get("dataset_kwargs", {})

    # Load the data
    data = load_data(data_path, is_local, **dataset_kwargs)

    # Format the data
    formatted_data = format_data(data, formatter_type, column_mapping)

    return formatted_data


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and format data from Hugging Face"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON or YAML)",
    )
    args = parser.parse_args()

    formatted_data = load_and_format_data(args.config)
    print(f"Loaded and formatted data: {len(formatted_data)} samples")
