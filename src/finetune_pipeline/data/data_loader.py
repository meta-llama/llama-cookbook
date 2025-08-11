"""
Data loader module for loading and formatting data from Hugging Face.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        Dataset object from the datasets library with all splits

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
        # Add image(s) to user content
        if image is not None:
            if isinstance(image, list):
                # Handle list of images
                for img in image:
                    if img:  # Check if image path is not empty
                        user_content.append(
                            {"type": "image", "image_url": {"url": img}}
                        )
            else:
                # Handle single image
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


def save_formatted_data(
    formatted_data: List[Any], output_dir: str, formatter_type: str, split: str
) -> str:
    """
    Save formatted data to a JSON file.

    Args:
        formatted_data: The formatted data to save
        output_dir: Directory to save the data
        formatter_type: Type of formatter used ('torchtune', 'vllm', or 'openai')

    Returns:
        Path to the saved file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    formatted_data_path = os.path.join(
        output_dir, f"{split}_{formatter_type}_formatted_data.json"
    )

    # Save the formatted data
    with open(formatted_data_path, "w") as f:
        # Handle different data types
        if isinstance(formatted_data, list) and all(
            isinstance(item, dict) for item in formatted_data
        ):
            json.dump(formatted_data, f, indent=2)
        elif isinstance(formatted_data, list) and all(
            isinstance(item, str) for item in formatted_data
        ):
            json.dump(formatted_data, f, indent=2)
        else:
            # For other types, convert to a simple list of strings
            json.dump([str(item) for item in formatted_data], f, indent=2)

    print(f"Saved formatted data to {formatted_data_path}")
    return formatted_data_path


def save_conversation_data(conversation_data: List, output_dir: str, split: str) -> str:
    """
    Save conversation data to a JSON file.

    Args:
        conversation_data: List of Conversation objects
        output_dir: Directory to save the data

    Returns:
        Path to the saved file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    conversation_data_path = os.path.join(output_dir, f"{split}_conversation_data.json")

    # Convert Conversation objects to a serializable format
    serializable_conversations = []
    for conv in conversation_data:
        serializable_conversations.append({"messages": conv.messages})

    # Save the conversation data
    with open(conversation_data_path, "w") as f:
        json.dump(serializable_conversations, f, indent=2)

    print(f"Saved conversation data to {conversation_data_path}")
    return conversation_data_path


def format_data(
    data,
    formatter_type: str,
    output_dir: str,
    column_mapping: Optional[Dict] = None,
    dataset_kwargs: Optional[Dict] = None,
):
    """
    Format the data using the specified formatter for all splits.

    Args:
        data: Dataset with multiple splits to format or a single dataset
        formatter_type: Type of formatter to use ('torchtune', 'vllm', or 'openai')
        output_dir: Directory to save the formatted data
        column_mapping: Optional mapping of column names
        dataset_kwargs: Optional dataset kwargs that may contain split information

    Returns:
        Tuple containing (formatted_data_paths, conversation_data_paths) where each is a list of paths to saved files
    """
    formatted_data_paths = []
    conversation_data_paths = []

    # Check if the dataset has explicit splits
    if (
        hasattr(data, "keys")
        and callable(data.keys)
        and len(data.keys()) > 0
        and isinstance(data, dict)
    ):
        # Dataset has splits (train, validation, test, etc.)
        splits = data.keys()

        for split in splits:
            # First convert the data to conversations
            conversations = convert_to_conversations(data[split], column_mapping)

            # Then get the formatter and format the conversations
            formatter = get_formatter(formatter_type)
            formatted_data = formatter.format_data(conversations)
            print(
                f"Loaded and formatted data for split '{split}': {len(formatted_data)} samples"
            )

            # Save the formatted data
            formatted_data_path = save_formatted_data(
                formatted_data, output_dir, formatter_type, split
            )
            formatted_data_paths.append(formatted_data_path)

            # Save the conversation data
            conversation_data_path = save_conversation_data(
                conversations, output_dir, split
            )
            conversation_data_paths.append(conversation_data_path)
    else:
        # Dataset doesn't have explicit splits, treat it as a single dataset
        # Check if a split is specified in dataset_kwargs
        split = "default"
        if dataset_kwargs and "split" in dataset_kwargs:
            split = dataset_kwargs["split"]

        # First convert the data to conversations
        conversations = convert_to_conversations(data, column_mapping)

        # Then get the formatter and format the conversations
        formatter = get_formatter(formatter_type)
        formatted_data = formatter.format_data(conversations)
        print(
            f"Loaded and formatted data for split '{split}': {len(formatted_data)} samples"
        )

        # Save the formatted data
        formatted_data_path = save_formatted_data(
            formatted_data, output_dir, formatter_type, split
        )
        formatted_data_paths.append(formatted_data_path)

        # Save the conversation data
        conversation_data_path = save_conversation_data(
            conversations, output_dir, split
        )
        conversation_data_paths.append(conversation_data_path)

    return formatted_data_paths, conversation_data_paths


def load_and_format_data(formatter_config: Dict, output_dir: str):
    """
    Load and format data based on the configuration.

    Args:
        formatter_config: Dictionary containing formatter configuration parameters
        output_dir: Directory to save the formatted data

    Returns:
        Tuple containing (formatted_data_paths, conversation_data_paths) where each is a list of paths to saved files
    """

    # Extract parameters from config
    data_path = formatter_config.get("data_path")
    if not data_path:
        raise ValueError(
            "data_path must be specified in the formatter section of the config file"
        )

    is_local = formatter_config.get("is_local", False)
    formatter_type = formatter_config.get("type", "torchtune")
    column_mapping = formatter_config.get("column_mapping")
    dataset_kwargs = formatter_config.get("dataset_kwargs", {})

    # Load the data
    data = load_data(data_path, is_local, **dataset_kwargs)

    # Format the data
    formatted_data_paths, conversation_data_paths = format_data(
        data, formatter_type, output_dir, column_mapping, dataset_kwargs
    )

    return formatted_data_paths, conversation_data_paths


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

    # Read the configuration
    config = read_config(args.config)
    formatter_config = config.get("formatter", {})
    output_dir = config.get("output_dir", "/tmp/finetune-pipeline/data/")
    output_data_dir = os.path.join(output_dir, "data")
    # Load and format the data
    formatted_data_paths, conversation_data_paths = load_and_format_data(
        formatter_config, output_data_dir
    )
