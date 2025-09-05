"""
Activation checkpointing utilities for single GPU training.
"""
import torch
from typing import Optional, List, Type
from transformers import PreTrainedModel
import warnings
import functools

# --- Improved Import Block ---
# Attempt to import layer classes for various models individually to provide specific warnings.
# This makes the manual checkpointing function more robust and informative.
TRANSFORMER_LAYER_CLASSES: List[Type[torch.nn.Module]] = []

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    TRANSFORMER_LAYER_CLASSES.append(LlamaDecoderLayer)
except ImportError:
    warnings.warn(
        "Could not import LlamaDecoderLayer. Manual activation checkpointing for Llama-like models will not be available."
    )

try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    TRANSFORMER_LAYER_CLASSES.append(MistralDecoderLayer)
except ImportError:
    warnings.warn(
        "Could not import MistralDecoderLayer. Manual activation checkpointing for Mistral-like models will not be available."
    )

try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
    TRANSFORMER_LAYER_CLASSES.append(GemmaDecoderLayer)
except ImportError:
    warnings.warn(
        "Could not import GemmaDecoderLayer. Manual activation checkpointing for Gemma-like models will not be available."
    )

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    TRANSFORMER_LAYER_CLASSES.append(Qwen2DecoderLayer)
except ImportError:
    warnings.warn(
        "Could not import Qwen2DecoderLayer. Manual activation checkpointing for Qwen2-like models will not be available."
    )
# --- End of Improved Import Block ---


def apply_activation_checkpointing(
    model: PreTrainedModel,
    use_reentrant: bool = False,
) -> PreTrainedModel:
    """
    Applies activation checkpointing to a model for memory-efficient training.
    This is the recommended function and uses the model's built-in Hugging Face implementation.
    
    Args:
        model: The model to apply checkpointing to (must be a PreTrainedModel).
        use_reentrant: Whether to use the reentrant implementation of checkpointing.
                       False is recommended as it's more memory-efficient.
    
    Returns:
        The model with activation checkpointing enabled.
    """
    if not hasattr(model, "gradient_checkpointing_enable"):
        warnings.warn(
            f"Model type {type(model).__name__} does not support gradient checkpointing. "
            "Activation checkpointing not applied."
        )
        return model

    # Use the official Hugging Face API to enable checkpointing
    try:
        # Try the modern API with gradient_checkpointing_kwargs
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    except TypeError:
        # Fallback for older transformers versions that don't have the kwargs
        model.gradient_checkpointing_enable()
        if use_reentrant is False: # Only warn if user explicitly requested the unsupported option
             warnings.warn(
                "Your version of `transformers` does not support the `use_reentrant` kwarg. "
                "Activation checkpointing has been enabled with the library's default behavior."
            )
    
    print(f"✓ Enabled activation checkpointing (use_reentrant={use_reentrant}) using the official API.")
    
    # Set a flag to indicate checkpointing is enabled
    model._activation_checkpointing_enabled = True
    
    return model


def _apply_activation_checkpointing_manual(
    model: PreTrainedModel,
    use_reentrant: bool = False,
    checkpoint_method: str = "uniform",
    checkpoint_layers: Optional[List[int]] = None
) -> PreTrainedModel:
    """
    (Internal/Advanced Use) Manual implementation of activation checkpointing.
    
    This function manually wraps decoder layers to apply checkpointing. It is more fragile
    than the primary `apply_activation_checkpointing` function but provides finer-grained
    control for advanced use cases or models not fully supported by the HF API.
    
    Args:
        model: The model to apply checkpointing to.
        use_reentrant: Whether to use reentrant checkpointing.
        checkpoint_method: Method for selecting layers ("uniform", "all", "manual").
        checkpoint_layers: Specific layer indices for "manual" method.
    
    Returns:
        The model with activation checkpointing enabled.
    """
    if not TRANSFORMER_LAYER_CLASSES:
        warnings.warn(
            "No supported transformer layer classes were found. Manual checkpointing cannot be applied."
        )
        return model

    # Store original forward methods if they haven't been stored already
    if not hasattr(model, "_original_forward_methods"):
        model._original_forward_methods = {}

    # Find all decoder layers that match the imported types
    decoder_layers = [
        (name, module) for name, module in model.named_modules() 
        if isinstance(module, tuple(TRANSFORMER_LAYER_CLASSES))
    ]

    if not decoder_layers:
        warnings.warn("Could not find any supported transformer decoder layers to checkpoint in this model.")
        return model

    # Determine which layers to checkpoint
    if checkpoint_method == "all":
        layers_to_checkpoint = list(range(len(decoder_layers)))
    elif checkpoint_method == "uniform":
        layers_to_checkpoint = list(range(0, len(decoder_layers), 2))
    elif checkpoint_method == "manual" and checkpoint_layers is not None:
        layers_to_checkpoint = checkpoint_layers
    else: # Default to uniform if method is invalid or manual is chosen without layers
        if checkpoint_method != "uniform":
            warnings.warn(f"Invalid checkpoint_method '{checkpoint_method}' or missing checkpoint_layers. Defaulting to 'uniform'.")
        layers_to_checkpoint = list(range(0, len(decoder_layers), 2))

    checkpointed_count = 0
    for i, (name, layer) in enumerate(decoder_layers):
        if i in layers_to_checkpoint:
            # Save the original forward method if not already saved
            if name not in model._original_forward_methods:
                model._original_forward_methods[name] = layer.forward
            
            # Wrap the forward method
            layer.forward = functools.partial(
                _checkpointed_forward,
                original_forward=model._original_forward_methods[name],
                use_reentrant=use_reentrant,
            )
            checkpointed_count += 1

    print(f"✓ Manually applied activation checkpointing to {checkpointed_count}/{len(decoder_layers)} layers using '{checkpoint_method}' method.")
    model._activation_checkpointing_enabled = True
    
    return model


def _checkpointed_forward(original_forward, *args, use_reentrant=False, **kwargs):
    """Helper function for the checkpointed forward pass used by the manual wrapper."""
    if torch.is_grad_enabled():  # More robust than checking `model.training`
        # Filter out None arguments which `torch.utils.checkpoint` doesn't handle well
        filtered_args = [arg for arg in args if arg is not None]
        return torch.utils.checkpoint.checkpoint(
            original_forward,
            *filtered_args,
            use_reentrant=use_reentrant,
            **kwargs
        )
    return original_forward(*args, **kwargs)


def disable_activation_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    """
    Disables activation checkpointing on a model, restoring its original state.
    Handles both the Hugging Face API and manual wrapper approaches.
    
    Args:
        model: The model to disable checkpointing on.
        
    Returns:
        The model with activation checkpointing disabled.
    """
    # 1. Disable using the official API (safe to call even if not enabled)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        # Only print if it was likely enabled this way
        if getattr(model, "_activation_checkpointing_enabled", False):
            print("✓ Disabled activation checkpointing via Hugging Face API.")

    # 2. Restore any manually patched methods
    if hasattr(model, "_original_forward_methods"):
        restored_count = 0
        for name, original_forward in model._original_forward_methods.items():
            try:
                # Recursively find the module by its fully qualified name and restore its forward method
                module = model.get_submodule(name)
                module.forward = original_forward
                restored_count += 1
            except AttributeError:
                warnings.warn(f"Could not find module '{name}' to restore its forward method.")
        
        if restored_count > 0:
            print(f"✓ Restored {restored_count} manually patched forward methods.")
        
        # Clean up the stored methods to leave the model in a clean state
        del model._original_forward_methods
    
    # Clear the tracking flag
    if hasattr(model, "_activation_checkpointing_enabled"):
        model._activation_checkpointing_enabled = False
    
    return model
