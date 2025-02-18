# VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/add_cot_vllm.py --model_id meta-llama/Llama-3.3-70B-Instruct --dataset-path datasets/2_ready_for_CoT/func-calling-multi-turn-final/ --config configs/config.yaml --output-path "datasets/3_CoT_added/func-calling-multi-turn-final/" --batch-size 96 --max-seq-len 16000


import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class LLM_Singleton:
    _instance = None

    def __new__(
        cls,
        model_id,
        max_model_len=64000,
        max_num_seqs=16,
        enforce_eager=True,
        debug=False,
    ):
        if cls._instance is None:
            cls._instance = super(LLM_Singleton, cls).__new__(cls)
            cls._instance._initialize(
                model_id,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager,
                debug=debug,
            )
        return cls._instance

    def _initialize(
        self,
        model_id,
        tensor_parallel_size=1,
        max_model_len=64000,
        max_num_seqs=16,
        enforce_eager=True,
        debug=False,
    ):
        if debug:
            print(
                f"Initializing LLM with params: {model_id}, {tensor_parallel_size}, {max_model_len}"
            )

        self.llm = LLM(
            model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.95,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)


def load_system_prompt(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config["system_prompt"]


def create_chat_message(system_prompt, conversation):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation},
    ]
    return messages


def parse_json_output(output_text):
    output_text = output_text.strip()
    json_match = re.search(r"\[.*\]", output_text, re.DOTALL)

    if json_match:
        output_text = json_match.group(0)

    try:
        if output_text.startswith('"') and output_text.endswith('"'):
            output_text = json.loads(output_text)
        result = json.loads(output_text)

        # Clean the result to remove 'tool': None entries
        cleaned_result = []
        for item in result:
            cleaned_item = {
                k: v for k, v in item.items() if k != "tool" or v is not None
            }
            cleaned_result.append(cleaned_item)

        return cleaned_result
    except json.JSONDecodeError as e:
        print(f"Error parsing output: {e}")
        return None


def process_dataset(
    dataset,
    system_prompt: str,
    start_index: int = 0,
    end_index: int = None,
    n_samples: int = 0,
    model_instance: Any = None,
    batch_size: int = 16,
    max_seq_len: int = 64000,
) -> List[Dict]:
    if end_index is None:
        end_index = len(dataset)
    else:
        end_index = min(end_index, len(dataset))

    # Handle random sampling
    dataset_size = end_index - start_index
    if n_samples > 0:
        n_samples = min(n_samples, dataset_size)
        indices = random.sample(range(start_index, end_index), n_samples)
        dataset_slice = dataset.select(indices)
    else:
        dataset_slice = dataset.select(range(start_index, end_index))

    results = []

    for i in tqdm(range(0, len(dataset_slice), batch_size), desc=f"Processing batches"):
        batch_slice = dataset_slice.select(
            range(i, min(i + batch_size, len(dataset_slice)))
        )

        try:
            batch_inputs = []
            for item in batch_slice:
                conversation_str = json.dumps(
                    item["conversations"], ensure_ascii=False, indent=2
                )
                messages = create_chat_message(system_prompt, conversation_str)
                input_text = model_instance.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                batch_inputs.append(input_text)

            sampling_params = SamplingParams(
                max_tokens=max_seq_len, temperature=0.1, top_p=0.95
            )

            outputs = model_instance.llm.generate(batch_inputs, sampling_params)

            for item, output in zip(batch_slice, outputs):
                enhanced_convos = parse_json_output(output.outputs[0].text.strip())
                if enhanced_convos is None:
                    print(
                        f"Warning: Failed to parse output for item {item.get('id', 'unknown')}"
                    )
                    enhanced_convos = item["conversations"]

                results.append(
                    {
                        "id": item["id"],
                        "conversations": item["conversations"],
                        "cot_conversations": enhanced_convos,
                    }
                )

        except Exception as e:
            print(f"Error processing batch starting at {i}: {str(e)}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset to enhance conversations with CoT reasoning"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config with system prompt",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output dataset directory path",
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Input dataset path"
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Starting index (inclusive)"
    )
    parser.add_argument("--end-index", type=int, help="Ending index (exclusive)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Number of random samples to process. If 0, process all samples in range",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for processing"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64000,
        help="Maximum sequence length for generation per batch",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Maximum number of sequences in batch",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Whether to enforce eager execution",
    )
    args = parser.parse_args()

    # Set spawn method for multiprocessing
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    system_prompt = load_system_prompt(args.config)
    dataset = load_from_disk(args.dataset_path)
    if isinstance(dataset, dict):
        dataset = dataset["train"]

    # Initialize VLLM instance
    model_instance = LLM_Singleton(
        model_id=args.model_id,
        max_model_len=args.max_seq_len,
        max_num_seqs=16,
        enforce_eager=True,
        debug=args.debug,
    )

    results = process_dataset(
        dataset=dataset,
        system_prompt=system_prompt,
        start_index=args.start_index,
        end_index=args.end_index,
        n_samples=args.n_samples,
        model_instance=model_instance,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    output_dataset = Dataset.from_dict(
        {
            "id": [r["id"] for r in results],
            "conversations": [r["conversations"] for r in results],
            "cot_conversations": [r["cot_conversations"] for r in results],
        }
    )

    output_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
