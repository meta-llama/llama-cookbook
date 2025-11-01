#!/usr/bin/env python3
"""
Script to modify the dataset by removing the top-level 'gt_parse' attribute from the ground_truth column
and keeping all the keys under it. Also supports custom train-test splits.
"""

import argparse
import json
import logging

from datasets import load_dataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare W2 dataset with custom train-test splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (default: 0.8, i.e., 80%% train, 20%% test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory name. If not provided, will use 'fake_w2_us_tax_form_dataset_train{train_ratio}_test{1 - train_ratio}'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Parse this W-2 form and extract all fields into a single level json.",
        help="Custom prompt to use for the input field (default: Parse this W-2 form...)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="singhsays/fake-w2-us-tax-form-dataset",
        help="Dataset name from HuggingFace Hub (default: singhsays/fake-w2-us-tax-form-dataset)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation split loading (useful if dataset doesn't have validation split)",
    )
    return parser.parse_args()


# Define a function to modify the ground_truth column
def remove_gt_parse_wrapper(example):
    try:
        # Parse the ground_truth JSON
        ground_truth = json.loads(example["ground_truth"])

        # Check if gt_parse exists in the ground_truth
        if "gt_parse" in ground_truth:
            # Replace the ground_truth with just the contents of gt_parse
            example["ground_truth"] = json.dumps(ground_truth["gt_parse"])
        else:
            logger.warning("No 'gt_parse' key found in ground_truth, keeping original")

        return example
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ground_truth JSON: {e}")
        logger.error(f"Problematic data: {example.get('ground_truth', 'N/A')}")
        # Return the example unchanged if we can't parse it
        return example
    except Exception as e:
        logger.error(f"Unexpected error in remove_gt_parse_wrapper: {e}")
        return example


def validate_dataset(dataset):
    """Validate the loaded dataset has required columns."""
    required_columns = ["ground_truth", "image"]
    missing_columns = [
        col for col in required_columns if col not in dataset.column_names
    ]

    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    logger.info(f"Dataset validation passed. Columns: {dataset.column_names}")


def validate_train_ratio(train_ratio):
    """Validate that train ratio is between 0 and 1 (exclusive)."""
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Train ratio must be between 0 and 1 (exclusive)")
    return True


def create_output_directory_name(train_ratio, test_ratio, output_dir=None):
    """Create output directory name based on the split ratio if not provided."""
    if output_dir is None:
        # Round to 2 decimal places before converting to int to avoid floating point precision issues
        train_pct = int(round(train_ratio * 100, 2))
        test_pct = int(round(test_ratio * 100, 2))
        return f"fake_w2_us_tax_form_dataset_train{train_pct}_test{test_pct}"
    return output_dir


def load_dataset_safely(dataset_name, split="train+test"):
    """Load dataset with proper error handling."""
    try:
        return load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise


def create_splits(all_data, train_ratio, seed):
    """Create train-test splits from the dataset."""
    logger.info(f"Creating new splits with train ratio: {train_ratio}")
    return all_data.train_test_split(train_size=train_ratio, seed=seed)


def load_validation_split(dataset_name, split_ds, skip_validation=False):
    """Load validation split if not skipped."""
    if skip_validation:
        logger.info("Skipping validation split as requested")
        return split_ds

    try:
        split_ds["validation"] = load_dataset(dataset_name, split="validation")
        logger.info(
            f"Loaded validation split with {len(split_ds['validation'])} examples"
        )
    except Exception as e:
        logger.warning(
            f"Could not load validation split: {e}. Continuing without validation split."
        )

    return split_ds


def apply_transformations(split_ds, prompt):
    """Apply data transformations to the dataset."""
    logger.info("Modifying dataset...")
    modified_ds = split_ds.map(remove_gt_parse_wrapper)

    logger.info(f"Adding custom prompt: {prompt}")
    modified_ds = modified_ds.map(lambda _: {"input": prompt})

    return modified_ds


def log_dataset_statistics(all_data, modified_ds):
    """Log comprehensive dataset statistics."""
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total examples: {len(all_data)}")
    logger.info(
        f"Train split: {len(modified_ds['train'])} examples ({len(modified_ds['train'])/len(all_data)*100:.1f}%)"
    )
    logger.info(
        f"Test split: {len(modified_ds['test'])} examples ({len(modified_ds['test'])/len(all_data)*100:.1f}%)"
    )
    if "validation" in modified_ds:
        logger.info(f"Validation split: {len(modified_ds['validation'])} examples")


def save_dataset(modified_ds, output_dir):
    """Save the modified dataset to disk."""
    logger.info(f"Saving modified dataset to '{output_dir}'...")
    modified_ds.save_to_disk(output_dir)
    logger.info(f"Done! Modified dataset saved to '{output_dir}'")


def main():
    try:
        args = parse_args()

        # Reconfigure logging with user-specified level
        global logger

        # Validate train ratio
        validate_train_ratio(args.train_ratio)

        train_ratio = args.train_ratio
        test_ratio = 1 - train_ratio

        # Create output directory name
        output_dir = create_output_directory_name(
            train_ratio, test_ratio, args.output_dir
        )

        logger.info(f"Using train-test split: {train_ratio:.2f}-{test_ratio:.2f}")
        logger.info(f"Output directory will be: {output_dir}")
        logger.info(f"Dataset: {args.dataset_name}")

        # Load the dataset with error handling
        logger.info("Loading dataset...")
        all_data = load_dataset_safely(args.dataset_name, "train+test")

        validate_dataset(all_data)
        logger.info(f"Loaded {len(all_data)} examples from dataset")

        # Create splits
        split_ds = create_splits(all_data, train_ratio, args.seed)

        # Load validation split
        split_ds = load_validation_split(
            args.dataset_name, split_ds, args.skip_validation
        )

        # Apply transformations
        modified_ds = apply_transformations(split_ds, args.prompt)

        # Log statistics
        log_dataset_statistics(all_data, modified_ds)

        # Save the modified dataset
        save_dataset(modified_ds, output_dir)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
