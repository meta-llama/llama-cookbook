import fire
import os
import yaml
import torch
from transformers import AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
from llama_recipes.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu


def main(
    fsdp_checkpoint_path: str = "",  # Path to FSDP Sharded model checkpoints
    consolidated_model_path: str = "",  # Path to save the HF converted model checkpoints
    HF_model_path_or_name: str = "",  # Path/ name of the HF model that includes config.json and tokenizer_config.json
    use_bfloat16: bool = True  # Whether to convert the model to bfloat16 precision
):
    """
    Convert FSDP sharded model checkpoints to Hugging Face format with optional bfloat16 conversion.

    Arguments:
    fsdp_checkpoint_path (str): Path to the FSDP sharded checkpoints (directory with .distcp files).
    consolidated_model_path (str): Path where the converted Hugging Face model will be saved.
    HF_model_path_or_name (str): Name or path to the Hugging Face model to load config/tokenizer (e.g., 'meta-llama/Llama-3.2-11B-Vision-Instruct').
    use_bfloat16 (bool): Flag to convert the model to bfloat16 precision during the process.

    Example:
        python3 Checkpoint_converter_llama-3-2-vision-fsdp_hf.py \
            --fsdp_checkpoint_path /path/to/fsdp/checkpoints \
            --consolidated_model_path /path/to/save/hf_model \
            --HF_model_path_or_name meta-llama/Llama-3.2-11B-Vision-Instruct \
            --use_bfloat16 True
    """
    try:
        # Attempt to load model name from train_params.yaml
        file_name = 'train_params.yaml'
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        with open(train_params_path, 'r') as file:
            data = yaml.safe_load(file)
            HF_model_path_or_name = data.get('model_name', HF_model_path_or_name)
            print(f"Model name from train_params.yaml: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"train_params.yaml not found in {fsdp_checkpoint_path}. Using provided model name.")
    except Exception as e:
        print(f"Error loading train_params.yaml: {e}")

    # Load the model definition from the Hugging Face model config using MllamaForConditionalGeneration
    model = MllamaForConditionalGeneration.from_pretrained(
        HF_model_path_or_name,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        device_map="auto"
    )
    print("Model loaded from Hugging Face config")

    # Load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model, fsdp_checkpoint_path)
    print("Model loaded from FSDP checkpoints")

    # Load and save the tokenizer from the Hugging Face model path
    tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)

    # Save the FSDP sharded checkpoints in Hugging Face format (bfloat16 if applicable)
    model.save_pretrained(consolidated_model_path, safe_serialization=True)
    print(f"Hugging Face model checkpoints have been saved in {consolidated_model_path}")

if __name__ == "__main__":
    fire.Fire(main)
