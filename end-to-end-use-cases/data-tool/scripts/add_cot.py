import argparse
import json
import os
from typing import Any, Dict, List, Union

import torch
import yaml
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def load_system_prompt(yaml_path: str) -> str:
    """Load system prompt from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config["system_prompt"]


def setup_llm(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 128000,
    gpu_ids: List[int] = None,
):
    """Initialize the vLLM LLM with specified parameters for multi-GPU support."""
    from vllm import LLM, SamplingParams

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    return llm, SamplingParams


def setup_hf_pipeline(
    model_name: str,
    gpu_ids: List[int] = None,
):
    """Initialize the HuggingFace pipeline."""
    import transformers

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def create_messages(system_prompt: str, conversation: str) -> List[Dict[str, str]]:
    """Create the messages list for the model input."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation},
    ]


def format_prompt(system_prompt: str, conversation: str) -> str:
    """Format the system prompt and conversation into the specific chat template format."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{conversation}"
    )


def process_with_vllm(
    item: Dict,
    llm: Any,
    system_prompt: str,
    sampling_params: Any,
) -> str:
    """Process a single item using vLLM."""
    prompt = format_prompt(system_prompt, item["conversations"])
    output = llm.generate(prompt, sampling_params)[0]
    return output.outputs[0].text


def process_with_hf(
    item: Dict,
    pipeline: Any,
    system_prompt: str,
    max_new_tokens: int,
) -> str:
    """Process a single item using HuggingFace pipeline."""
    messages = create_messages(system_prompt, item["conversations"])
    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
    )
    return outputs[0]["generated_text"][-1]["content"]


def process_dataset(
    dataset,
    system_prompt: str,
    output_file: str,
    start_index: int = 0,
    end_index: int = None,
    max_new_tokens: int = 128000,
    use_hf: bool = False,
    model_instance: Any = None,
    sampling_params: Any = None,
) -> None:
    """Process the dataset using either vLLM or HuggingFace pipeline."""
    # Handle end_index
    if end_index is None:
        end_index = len(dataset)
    else:
        end_index = min(end_index, len(dataset))

    # Validate indices
    if start_index < 0:
        start_index = 0
    if start_index >= len(dataset):
        raise ValueError(
            f"Start index {start_index} is larger than dataset size {len(dataset)}"
        )
    if start_index >= end_index:
        raise ValueError(
            f"Start index {start_index} must be less than end index {end_index}"
        )

    # Select the specified range
    dataset_slice = dataset.select(range(start_index, end_index))

    # Process examples one at a time
    with open(output_file, "w") as f:
        for item in tqdm(
            dataset_slice, desc=f"Processing rows {start_index} to {end_index}"
        ):
            # Generate the response using appropriate method
            if use_hf:
                cot_response = process_with_hf(
                    item, model_instance, system_prompt, max_new_tokens
                )
            else:
                cot_response = process_with_vllm(
                    item, model_instance, system_prompt, sampling_params
                )

            # Save the result with both original and CoT conversations
            result = {
                "id": item["id"],
                "conversations": item["conversations"],  # Keep original conversations
                "cot_conversations": cot_response,  # Add new CoT conversations
            }
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset using vLLM or HuggingFace pipeline with multi-GPU support"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name or path of the model to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file containing system prompt",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="processed_outputs.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in the dataset (inclusive)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Ending index in the dataset (exclusive). If not specified, processes until the end.",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use HuggingFace pipeline instead of vLLM",
    )
    args = parser.parse_args()

    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]

    # Load system prompt from YAML
    system_prompt = load_system_prompt(args.config)

    # Load dataset
    dataset = load_from_disk(args.dataset_path)

    # Initialize appropriate model instance based on mode
    sampling_params = None
    if args.use_hf:
        model_instance = setup_hf_pipeline(
            model_name=args.model,
            gpu_ids=gpu_ids,
        )
    else:
        model_instance, sampling_params = setup_llm(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            gpu_ids=gpu_ids,
        )
        sampling_params = sampling_params(
            max_tokens=128000,
            temperature=0.7,
            top_p=0.95,
        )

    # Process dataset
    process_dataset(
        dataset=dataset,
        system_prompt=system_prompt,
        output_file=args.output_file,
        start_index=args.start_index,
        end_index=args.end_index,
        use_hf=args.use_hf,
        model_instance=model_instance,
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    main()
