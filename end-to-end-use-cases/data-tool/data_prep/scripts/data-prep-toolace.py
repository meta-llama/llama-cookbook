#!/usr/bin/env python3
import argparse
import uuid

from datasets import Dataset, load_dataset
from tqdm import tqdm


def transform_dataset(dataset_name, output_path):
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Loaded dataset with {len(dataset['train'])} examples")

    new_data = {"id": [], "conversations": []}

    print("Transforming dataset...")
    for example in tqdm(dataset["train"], desc="Processing examples"):
        new_data["id"].append(str(uuid.uuid4()))

        transformed_conv = [{"from": "system", "value": example["system"]}] + example[
            "conversations"
        ]

        new_data["conversations"].append(transformed_conv)

    print("Creating new dataset...")
    new_dataset = Dataset.from_dict(new_data)

    print(f"Saving dataset to: {output_path}")
    new_dataset.save_to_disk(output_path)

    print(f"Successfully transformed {len(new_dataset)} examples")
    return new_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Transform dataset to conversation format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Team-ACE/ToolACE",
        help="HuggingFace dataset name (default: Team-ACE/ToolACE)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transformed_dataset",
        help="Output path for transformed dataset (default: transformed_dataset)",
    )

    args = parser.parse_args()

    transform_dataset(dataset_name=args.dataset, output_path=args.output)


if __name__ == "__main__":
    main()
