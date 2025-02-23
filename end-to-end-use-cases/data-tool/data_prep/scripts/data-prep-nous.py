# python data-prep-nous.py --type singleturn ~/task_datasets/1_Downloaded/hermes-function-calling-v1/func-calling-singleturn.json ~/task_datasets/2_Prepped_for_CoT/hermes-function-calling-v1/  --second-input ~/task_datasets/1_Downloaded/hermes-function-calling-v1/json-mode-agentic.json --target-size 150

# python data-prep-nous.py ~/task_datasets/1_Downloaded/hermes-function-calling-v1/glaive-function-calling-5k.json ~/task_datasets/2_Prepped_for_CoT/hermes-function-calling-v1/glaive-balanced --type glaive --target-size 500

# python data-prep-nous.py ~/task_datasets/1_Downloaded/hermes-function-calling-v1/json-mode-agentic.json ~/task_datasets/2_Prepped_for_CoT/balanced-json-modeagentic --type agentic --target-size 25

# python data-prep-nous.py ~/task_datasets/1_Downloaded/hermes-function-calling-v1/func-calling.json ~/task_datasets/2_Prepped_for_CoT/balanced_func_calling --type func --target-size 25

import argparse
import math
import os
import random
from collections import defaultdict

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# Category mappings
AGENTIC_CATEGORY_MAPPING = {
    "Simulacrum Agent": "Simulacra Agents",
    "Simulacra Agent": "Simulacra Agents",
    "Outlines Agents": "Outlines Agents",
    "Outlines Agent": "Outlines Agents",
    "Minecraft Agent": "Minecraft Agents",
    "Voyager MineCraft Agent": "Minecraft Agents",
    "Agent Frameworks": "Development Frameworks",
    "Copilot Frameworks": "Development Frameworks",
    "AI Analysis Agent": "Utility Agents",
    "Code Analysis Agent": "Utility Agents",
    "File Management Agent": "Utility Agents",
    "Utility Function": "Utility Agents",
    "WebBrowser Agent": "Utility Agents",
    "Data Structures": "Data Processing Agents",
    "Data Structure": "Data Processing Agents",
    "Data Compression": "Data Processing Agents",
    "DSPy Agents": "DSPy Agents",
    "LLM Agents": "LLM Agents",
    "Instructor Agents": "Instructor Agents",
    "Autogen Agents": "Autogen Agents",
    "LlamaIndex Agents": "LlamaIndex Agents",
    "Langchain Agents": "Langchain Agents",
}

GLAIVE_CATEGORY_MAPPING = {
    "Technology": "tech_computing",
    "Programming Concepts": "tech_computing",
    "Programming and Computer Science Questions": "tech_computing",
    "Web Development and Design": "tech_computing",
    "Database and SQL": "tech_computing",
    "Swift Programming": "tech_computing",
    "Cybersecurity and Encryption": "tech_computing",
    "Data Science": "data_analytics",
    "Data Analysis and Programming": "data_analytics",
    "Machine Learning": "data_analytics",
    "Natural Language Processing": "data_analytics",
    "Stocks and Orders": "finance_business",
    "Loan and Financial Calculations": "finance_business",
    "Finance & Economics": "finance_business",
    "Business Strategies": "finance_business",
    "Science Education": "science_education",
    "Science and Nature Exploration": "science_education",
    "Quantum Physics": "science_education",
    "Climate and Environmental Solutions": "science_education",
    "Flight Services": "services_productivity",
    "Location Services": "services_productivity",
    "Productivity": "services_productivity",
    "Request Management": "services_productivity",
    "History and Culture": "knowledge_culture",
    "Book Search": "knowledge_culture",
    "Literary Analysis": "knowledge_culture",
    "Language and Linguistics": "knowledge_culture",
    "Language and Logic": "knowledge_culture",
}

DEFAULT_CATEGORY = "Other"


def analyze_distribution(data, category_mapping):
    category_counts = defaultdict(int)

    for item in data:
        category = item["category"]
        if category_mapping:
            category = category_mapping.get(category, DEFAULT_CATEGORY)
        category_counts[category] += 1

    df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
    df["Percentage"] = df["Count"] / len(data) * 100
    return df.sort_values("Count", ascending=False)


def balance_dataset(data, target_size=25, category_mapping=None):
    category_groups = defaultdict(list)
    for item in data:
        original_category = item["category"]
        mapped_category = original_category
        if category_mapping:
            mapped_category = category_mapping.get(original_category, DEFAULT_CATEGORY)
        category_groups[mapped_category].append(item)

    print("\nOriginal distribution after category mapping:")
    for cat, items in category_groups.items():
        print(f"{cat}: {len(items)}")

    # Thanos
    balanced_data = []
    for category, items in category_groups.items():
        if len(items) > target_size:
            sampled_items = random.sample(items, target_size)
            balanced_data.extend(sampled_items)
        else:
            balanced_data.extend(items)

        if category_mapping:
            for item in balanced_data[-len(items) :]:
                item["category"] = category

    print(f"\nOriginal dataset size: {len(data)}")
    print(f"Balanced dataset size: {len(balanced_data)}")
    final_dist = analyze_distribution(balanced_data)
    print("\nFinal distribution:")
    print(final_dist)

    return balanced_data


def merge_singleturn_datasets(func_path, json_path, target_per_dataset=150):
    print("\nMerging single-turn datasets...")
    func_single = load_dataset("json", data_files=func_path)
    json_single = load_dataset("json", data_files=json_path)

    print(f"Original func_single size: {len(func_single['train'])}")
    print(f"Original json_single size: {len(json_single['train'])}")

    def downsample_and_tag(dataset, source_name, target_total):
        category_groups = defaultdict(list)
        for item in dataset["train"]:
            category_groups[item["category"]].append(item)

        num_categories = len(category_groups)
        samples_per_category = max(1, math.floor(target_total / num_categories))

        print(f"\n{source_name}:")
        print(f"Number of categories: {num_categories}")
        print(f"Samples per category: {samples_per_category}")

        balanced_data = []
        for category, items in category_groups.items():
            if len(items) > samples_per_category:
                sampled_items = random.sample(items, samples_per_category)
                balanced_data.extend(sampled_items)
            else:
                balanced_data.extend(items)

        for item in balanced_data:
            item["dataset_source"] = source_name

        return balanced_data

    func_balanced = downsample_and_tag(
        func_single, "func_calling_singleturn", target_per_dataset
    )
    json_balanced = downsample_and_tag(
        json_single, "json_mode_singleturn", target_per_dataset
    )

    merged_data = func_balanced + json_balanced

    print("\nFinal merged dataset statistics:")
    print(f"Total examples: {len(merged_data)}")
    print(f"From func_calling_singleturn: {len(func_balanced)}")
    print(f"From json_mode_singleturn: {len(json_balanced)}")

    return merged_data


def process_dataset(
    input_path, output_path, dataset_type, target_size=25, second_input_path=None
):
    print(f"\nProcessing dataset: {input_path}")
    print(f"Dataset type: {dataset_type}")

    if dataset_type == "singleturn" and second_input_path:
        data = merge_singleturn_datasets(input_path, second_input_path, target_size)
        balanced_data = data  # Done earlier
    else:
        dataset = load_dataset("json", data_files=input_path)

    category_mapping = None
    if dataset_type == "agentic":
        category_mapping = AGENTIC_CATEGORY_MAPPING
    elif dataset_type == "glaive":
        category_mapping = GLAIVE_CATEGORY_MAPPING

    balanced_data = balance_dataset(
        dataset["train"], target_size=target_size, category_mapping=category_mapping
    )
    balanced_dataset = Dataset.from_list(balanced_data)
    dataset_dict = DatasetDict({"train": balanced_dataset})
    dataset_dict.save_to_disk(output_path)
    print(f"\nSaved balanced dataset to {output_path}")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Process and balance datasets")
    parser.add_argument("input_path", help="Path to input JSON dataset")
    parser.add_argument("output_path", help="Path to save balanced dataset")
    parser.add_argument(
        "--type",
        choices=["agentic", "func", "singleturn", "glaive"],
        required=True,
        help="Type of dataset to process",
    )
    parser.add_argument(
        "--second-input",
        help="Second input path (required for singleturn merge)",
        default=None,
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=25,
        help="Target size per category (default: 25)",
    )

    args = parser.parse_args()
    process_dataset(args.input_path, args.output_path, args.type, args.target_size)


if __name__ == "__main__":
    main()
