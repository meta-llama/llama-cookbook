"""
Data loader module for loading and formatting data from Hugging Face.
"""

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from PIL import Image

# Try to import yaml, but don't fail if it's not available
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Try to import datasets, but don't fail if it's not available
try:
    from datasets import Dataset, load_dataset, load_from_disk

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

    # Define dummy functions to avoid "possibly unbound" errors
    def load_dataset(*args, **kwargs):
        raise ImportError("The 'datasets' package is required to load data.")

    def load_from_disk(*args, **kwargs):
        raise ImportError("The 'datasets' package is required to load data.")


def image_to_base64(image: Union[str, Image.Image]):
    if isinstance(image, str):
        with open(image, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        return base64.b64encode(image.tobytes()).decode("utf-8")


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


def load_dataset(
    data_path: str,
    is_local: bool = False,
    column_mapping: Optional[Dict] = None,
    **kwargs,
):
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

    dataset = None
    if is_local:
        # Load from local disk
        file_extension = Path(data_path).suffix.lower()
        if file_extension in [".csv"]:
            data = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(data)
        else:
            dataset = load_from_disk(data_path)
    else:
        # Load from Hugging Face Hub
        dataset = load_dataset(data_path, **kwargs)

    # Rename columns if column_mapping is provided
    if column_mapping is None:
        column_mapping = {"input": "input", "output": "output", "image": "image"}

    required_fields = ["input", "output"]
    for field in required_fields:
        if field not in column_mapping:
            raise ValueError(f"Column mapping must include '{field}' field")
    dataset = dataset.rename_columns(column_mapping)

    return dataset


def convert_to_encoded_messages(
    example: Dict, system_prompt: Optional[str] = None
) -> Dict:
    image_field = "image"
    input_field = "input"
    output_field = "output"

    image = example.get(image_field, None)  # if image_field in example else None
    input_text = example.get(input_field, "")
    output_label = example.get(output_field, "")

    messages = []

    # Create system message if system_prompt is provided
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    # Create user content and user message
    user_content = [
        {"type": "text", "text": input_text},
    ]
    # Add image(s) to user content
    if image is not None:
        if not isinstance(image, list):
            image = [image]
        for img in image:
            b64_img = image_to_base64(img)
            user_content.append({"type": "image_url", "image_url": {"url": b64_img}})

    messages.append({"role": "user", "content": user_content})

    # Create assistant message with text content
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": output_label}],
        }
    )
    # Serialize to string and return. This is required because datasets.map adds extra keys to each dict in messages
    example["messages"] = json.dumps(messages)
    return example


def save_encoded_dataset(encoded_dataset, output_dir: str, split: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    conversation_data_path = os.path.join(output_dir, f"{split}_conversation_data.json")

    if not "messages" in encoded_dataset.column_names:
        raise RuntimeError
    messages = [json.loads(x) for x in encoded_dataset["messages"]]
    with open(conversation_data_path, "w") as f:
        json.dump(messages, f, indent=2)


# TODO: Verify if this is actually needed?
# def format_encoded_dataset(encoded_dataset, output_dir, split, format):
#     if format == "vllm":
#         messages = [json.loads(x) for x in encoded_dataset["messages"]]


def get_splits(dataset):
    """
    Helper function to get splits from a dataset.

    Args:
        dataset: HuggingFace dataset object

    Returns:
        List of split names
    """
    if hasattr(dataset, "keys"):
        return {k: dataset[k] for k in dataset.keys()}
    return {"default": dataset}


def get_hf_dataset(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    is_local: bool = False,
    column_mapping: Optional[Dict] = None,
    dataset_kwargs: Optional[Dict] = None,
    system_prompt: Optional[str] = None,
):
    """
    Load and format data based on either a configuration file or individual parameters.

    Args:
        output_dir: Directory to save the formatted data
        config_path: Path to configuration file (YAML/JSON). If provided, other parameters are ignored.
        dataset_id: Path/ID to the dataset to load
        is_local: Whether the data is stored locally
        column_mapping: Dictionary mapping column names
        dataset_kwargs: Additional arguments to pass to load_dataset
        system_prompt: System prompt to use for the dataset

    Returns:
        str: Path to the output directory containing the formatted data
    """

    # If config_path is provided, load from config file
    if config_path:
        config = read_config(config_path)
        output_dir = config.get("output_dir", "/tmp/finetuning-pipeline/outputs")
        data_config = config.get("data", {})

        # Extract parameters from data config
        dataset_id = data_config.get("dataset_id")
        is_local = data_config.get("is_local", False)
        column_mapping = data_config.get("column_mapping")
        dataset_kwargs = data_config.get("dataset_kwargs", {})
        system_prompt = data_config.get("system_prompt", None)
    else:
        # Use individual parameters passed to the function
        if dataset_kwargs is None:
            dataset_kwargs = {}

    # Validate required parameters
    if not dataset_id:
        raise ValueError(
            "dataset_id must be specified either in config file or as parameter"
        )

    # Load the dataset
    dataset = load_dataset(
        data_path=dataset_id,
        is_local=is_local,
        column_mapping=column_mapping,
        **dataset_kwargs,
    )

    # Get available splits
    dataset_splits = get_splits(dataset)

    # Process each split
    for split_name, split_dataset in dataset_splits.items():
        # Apply the conversion function
        encoded_dataset = split_dataset.map(
            lambda example: convert_to_encoded_messages(example, system_prompt)
        )

        # Save the encoded dataset
        save_encoded_dataset(encoded_dataset, output_dir, split_name)

        # TODO: Evaluate if formatting is needed here

    return output_dir


def main():
    """
    Example command-line interface for get_hf_dataset function.
    Shows how to use the function with either config file or individual arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Load and format HuggingFace dataset")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save formatted data",
        default="/tmp/finetuning-pipeline/outputs",
    )

    # Config file option
    parser.add_argument("--config", help="Path to config file (YAML/JSON)")

    # Individual parameter options
    parser.add_argument("--dataset_id", help="HF Dataset ID or path")
    parser.add_argument(
        "--is_local", action="store_true", help="Dataset is stored locally"
    )
    parser.add_argument("--system_prompt", help="System prompt for the dataset")
    parser.add_argument(
        "--split", help="Dataset split to load (e.g., train, validation)"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")

    args = parser.parse_args()

    if args.config:
        # Use config file
        print(f"Loading dataset using config file: {args.config}")
        result = get_hf_dataset(output_dir=args.output_dir, config_path=args.config)
    else:
        # Use individual arguments
        if not args.dataset_id:
            raise ValueError("--dataset_id is required when not using --config")

        dataset_kwargs = {}
        if args.split:
            dataset_kwargs["split"] = args.split
        if args.shuffle:
            dataset_kwargs["shuffle"] = args.shuffle

        print(f"Loading dataset using individual arguments: {args.dataset_id}")
        result = get_hf_dataset(
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            is_local=args.is_local,
            dataset_kwargs=dataset_kwargs,
            system_prompt=args.system_prompt,
        )

    print(f"Dataset processed and saved to: {result}")


if __name__ == "__main__":
    main()
