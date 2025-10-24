#!/usr/bin/env python3
"""
Example script for single GPU training with activation checkpointing.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

from llama_cookbook.utils.activation_checkpointing import apply_activation_checkpointing
from llama_cookbook.utils.memory_utils import (
    print_memory_stats, clear_memory, get_memory_stats
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def demonstrate_activation_checkpointing():
    """Demonstrate activation checkpointing with a small model."""
    
    print("=== Activation Checkpointing Demo ===\n")
    
    # Model selection based on available resources
    if torch.cuda.is_available():
        model_name = "gpt2"  # Using smaller model for demo
        device = "cuda"
        dtype = torch.float16
    else:
        model_name = "gpt2"  # Small model for CPU
        device = "cpu"
        dtype = torch.float32
    
    print(f"Using model: {model_name}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None
    )
    
    if device == "cuda":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print initial memory
    print_memory_stats("After model loading", detailed=True)
    
    # Apply activation checkpointing
    print("\nApplying activation checkpointing...")
    model = apply_activation_checkpointing(model, use_reentrant=False)
    
    # Prepare sample input
    text = "The future of AI is"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Test generation without training (inference)
    print("\nTesting inference...")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    
    # Clear memory and track training step
    clear_memory()
    print_memory_stats("Before training step", detailed=True)
    
    # Forward pass with labels for loss computation
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print_memory_stats("After backward pass", detailed=True)
    
    # Show memory savings
    stats = get_memory_stats()
    if 'gpu_allocated_gb' in stats:
        print(f"\n✓ Successfully demonstrated activation checkpointing!")
        print(f"  Current GPU memory usage: {stats['gpu_allocated_gb']:.2f}GB")
    else:
        print(f"\n✓ Successfully demonstrated activation checkpointing on CPU!")


if __name__ == "__main__":
    demonstrate_activation_checkpointing()
