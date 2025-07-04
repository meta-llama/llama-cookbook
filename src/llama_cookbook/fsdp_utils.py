# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
from typing import Any

def dequantize_model_for_fsdp(model: Any) -> bool:
    """
    Convert quantized model to bfloat16 for FSDP compatibility.
    
    Args:
        model: The PyTorch model to dequantize
        
    Returns:
        bool: True if dequantization was performed, False otherwise
    """
    # Check if model has quantized weights
    has_quantized_weights = any(
        p.dtype in [torch.int8, torch.uint8] 
        for p in model.parameters()
    )
    
    if has_quantized_weights:
        print("Converting quantized model to bfloat16 for FSDP compatibility...")
        
        # Convert all parameters to bfloat16
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        
        # Also convert buffers (like LayerNorm weights)
        for buffer_name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(torch.bfloat16)
            
        return True
    
    return False
