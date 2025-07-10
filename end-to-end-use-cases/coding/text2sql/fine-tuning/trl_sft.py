# Unified script supporting multiple fine-tuning configurations

import argparse
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Unified fine-tuning script for Llama 3.1 8B"
)
parser.add_argument(
    "--quantized",
    type=str,
    choices=["true", "false"],
    required=True,
    help="Whether to use quantization (true/false)",
)
parser.add_argument(
    "--peft",
    type=str,
    choices=["true", "false"],
    required=True,
    help="Whether to use PEFT (true/false)",
)
parser.add_argument(
    "--cot",
    type=str,
    choices=["true", "false"],
    required=True,
    help="Whether to use Chain-of-Thought dataset (true/false)",
)
args = parser.parse_args()

# Convert string arguments to boolean
use_quantized = args.quantized.lower() == "true"
use_peft = args.peft.lower() == "true"
use_cot = args.cot.lower() == "true"

# Check for unsupported combination
if not use_peft and use_quantized:
    print(
        "ERROR: Full Fine-Tuning (peft=false) with Quantization (quantized=true) is NOT RECOMMENDED!"
    )
    print("This combination can lead to:")
    print("- Gradient precision loss due to quantization")
    print("- Training instability")
    print("- Suboptimal convergence")
    print("\nRecommended combinations:")
    print(
        "1. --peft=true --quantized=true   (PEFT + Quantized - Most memory efficient)"
    )
    print("2. --peft=true --quantized=false  (PEFT + Non-quantized - Good balance)")
    print(
        "3. --peft=false --quantized=false (FFT + Non-quantized - Maximum performance)"
    )
    sys.exit(1)

print(f"Configuration: PEFT={use_peft}, Quantized={use_quantized}, CoT={use_cot}")

# Import additional modules based on configuration
if use_quantized:
    from transformers import BitsAndBytesConfig
if use_peft:
    from peft import LoraConfig

from trl import setup_chat_format, SFTConfig, SFTTrainer

# Dataset configuration based on CoT parameter
if use_cot:
    FT_DATASET = "train_text2sql_cot_dataset.json"
    print("Using Chain-of-Thought reasoning dataset")
else:
    FT_DATASET = "train_text2sql_sft_dataset.json"
    print("Using standard SFT dataset")

dataset = load_dataset("json", data_files=FT_DATASET, split="train")

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Configure quantization if needed
quantization_config = None
if use_quantized:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

# Configure PEFT if needed
peft_config = None
if use_peft:
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

# Configure training arguments based on combination using newer TRL API
cot_suffix = "cot" if use_cot else "nocot"

if use_peft and use_quantized:
    # PEFT + Quantized: Use SFTConfig (newer API)
    print("Using PEFT + Quantized configuration")
    args = SFTConfig(
        output_dir=f"llama31-8b-text2sql-peft-quantized-{cot_suffix}",
        num_train_epochs=3,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="tensorboard",
        max_seq_length=4096,
        packing=True,
    )

elif use_peft and not use_quantized:
    # PEFT + Non-quantized: Use SFTConfig (newer API)
    print("Using PEFT + Non-quantized configuration")
    args = SFTConfig(
        output_dir=f"llama31-8b-text2sql-peft-nonquantized-{cot_suffix}",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Slightly reduced for non-quantized
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="tensorboard",
        max_seq_length=4096,
        packing=True,
    )

else:  # not use_peft and not use_quantized
    # FFT + Non-quantized: Use SFTConfig (newer API)
    print("Using Full Fine-Tuning + Non-quantized configuration")
    args = SFTConfig(
        output_dir=f"llama31-8b-text2sql-fft-nonquantized-{cot_suffix}",
        num_train_epochs=1,  # Reduced epochs for full fine-tuning
        per_device_train_batch_size=1,  # Reduced batch size for full model training
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=5e-6,  # Lower learning rate for full fine-tuning
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,  # Standard gradient clipping for full fine-tuning
        warmup_ratio=0.1,  # Warmup ratio for full fine-tuning
        lr_scheduler_type="cosine",  # Cosine scheduler for full fine-tuning
        push_to_hub=True,
        report_to="tensorboard",
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        remove_unused_columns=False,  # Keep all columns
        max_seq_length=4096,
        packing=True,
    )

# Create trainer with consistent newer API
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Print memory requirements estimate
print("\nEstimated GPU Memory Requirements:")
if use_peft and use_quantized:
    print("- PEFT + Quantized: ~12-16 GB")
elif use_peft and not use_quantized:
    print("- PEFT + Non-quantized: ~20-25 GB")
else:  # FFT + Non-quantized
    print("- Full Fine-Tuning + Non-quantized: ~70-90 GB")

print("\nStarting training...")
trainer.train()

print("Training completed. Saving model...")
trainer.save_model()

print("Model saved successfully!")
