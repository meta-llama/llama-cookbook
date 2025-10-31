"""
Data loader module for loading and formatting data from Hugging Face.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from datasets import concatenate_datasets, load_dataset, load_from_disk

from ..utils import image_to_base64_url, load_config


def load_hf_dataset(data_path: str, is_local: bool = False, **kwargs):
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
    if not data_path:
        raise ValueError("data_path must be provided")

    dataset = None
    if is_local:
        # Load from local disk
        file_extension = Path(data_path).suffix.lower()
        if file_extension in [".csv"]:
            dataset = load_dataset("csv", data_files=data_path, **kwargs)
        else:
            dataset = load_from_disk(data_path, **kwargs)
    else:
        # Load from Hugging Face Hub
        dataset = load_dataset(data_path, **kwargs)

    return dataset


def resplit_dataset(dataset, train_percent: float, sample_percent: float = 1.0):
    if isinstance(dataset, dict):
        dataset = concatenate_datasets(list(dataset.values()))

    # sample
    dataset = dataset.take(int(len(dataset) * sample_percent))

    # resplit into "train" and "test" splits
    if train_percent == 0.0:
        # hf datasets doesn't allow empty splits; this will create a singleton split
        splits = dataset.train_test_split(train_size=1)
    elif train_percent == 1.0:
        splits = dataset.train_test_split(test_size=1)
    else:
        splits = dataset.train_test_split(train_size=train_percent)
    return splits


def convert_to_encoded_messages(
    example: Dict,
    image_column: str = None,
    user_text_column: str = None,
    assistant_text_column: str = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict:
    image = example.get(image_column, None)  # if image_field in example else None
    user_text = example.get(user_text_column, "")
    assistant_text = example.get(assistant_text_column, "")

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
    user_content = []
    if any([user_prompt, user_text]):
        user_text = user_prompt + "\n" + user_text
        user_content.append(
            {"type": "text", "text": user_text},
        )

    # Add image(s) to user content
    if image is not None:
        if not isinstance(image, list):
            image = [image]
        for img in image:
            b64_img_url = image_to_base64_url(img)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": b64_img_url},
                }
            )

    messages.append({"role": "user", "content": user_content})

    # Create assistant message with text content
    if assistant_text:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
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


def convert_hf_dataset(
    dataset: str,
    output_dir: str,
    is_local: bool = False,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    image_column: Optional[str] = None,
    user_text_column: Optional[str] = None,
    assistant_text_column: Optional[str] = None,
    resplit_train_percent: float = 0.7,
    sample_percent: float = 1.0,
):
    """
    Load and format data based on either a configuration file or individual parameters.

    Args:
        output_dir: Directory to save the formatted data
        dataset: Path/ID to the dataset to load
        is_local: Whether the data is stored locally
        system_prompt: System prompt to use for the dataset
        user_prompt: User prompt to prepend to user text
        image_column: Name of the column containing images
        user_text_column: Name of the column containing user text
        assistant_text_column: Name of the column containing assistant responses
        resplit_train_percent: Percentage of data to use for training split
        sample_percent: Percentage of total data to sample

    Returns:
        str: Path to the output directory containing the formatted data
    """

    # Load the dataset
    dataset = load_hf_dataset(data_path=dataset, is_local=is_local)

    # Concatenate and resplit the dataset into 'train' and 'test' splits
    dataset_splits = resplit_dataset(
        dataset, resplit_train_percent, sample_percent=sample_percent
    )

    # Process each split
    for split_name, split_dataset in dataset_splits.items():
        # Apply the conversion function with all parameters
        encoded_dataset = split_dataset.map(
            lambda example: convert_to_encoded_messages(
                example,
                image_column=image_column,
                user_text_column=user_text_column,
                assistant_text_column=assistant_text_column,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )

        # Save the encoded dataset
        save_encoded_dataset(encoded_dataset, output_dir, split_name)

    return output_dir


def run_dataset_builder(experiment_dir: str):
    script_dir = Path(__file__).parent.parent.parent
    config_path = script_dir / "config.yaml"

    # Load configuration
    config = load_config(config_path)

    # Set output directory
    output_dir = Path(experiment_dir) / "formatted_datasets"

    # get task1 dataset
    for task in ["task1", "task2"]:
        convert_hf_dataset(
            dataset=config[task].get("dataset"),
            output_dir=output_dir / task,
            is_local=config[task].get("is_local"),
            system_prompt=config[task].get("system_prompt"),
            user_prompt=config[task].get("user_prompt"),
            image_column=config[task].get("image_column"),
            user_text_column=config[task].get("user_text_column"),
            assistant_text_column=config[task].get("assistant_text_column"),
            resplit_train_percent=config[task].get("resplit_train_percent", None),
            sample_percent=config[task].get("sample_percent", 1.0),
        )


if __name__ == "__main__":
    run_dataset_builder(
        "/data/users/subramen/fbsource/fbcode/users/subramen/internal-llama-cookbook/end-to-end-use-cases/transferability/experiments/test01"
    )
