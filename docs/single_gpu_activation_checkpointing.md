# Single GPU Activation Checkpointing

## Overview

Activation checkpointing (gradient checkpointing) is a memory optimization technique that trades computation for memory. Instead of storing all intermediate activations during the forward pass, it recomputes them during the backward pass, significantly reducing memory usage.

## Implementation

This implementation leverages Hugging Face Transformers' built-in gradient checkpointing functionality, ensuring compatibility and optimal performance across different model architectures.

## Benefits

- **Memory Reduction**: 30-40% reduction in activation memory usage
- **Larger Batch Sizes**: Enables 50-70% larger batch sizes
- **Better GPU Utilization**: Higher throughput despite slower per-step training
- **Simple Integration**: Uses the model's native gradient checkpointing support

## Usage

### Basic Usage

Enable activation checkpointing for single GPU training:

```bash
python -m llama_cookbook.finetuning \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --enable_activation_checkpointing \
    --batch_size_training 4 \
    --dataset alpaca_dataset \
    --output_dir ./output
