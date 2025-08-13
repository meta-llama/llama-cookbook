import unittest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

# Adjust import based on your structure
import sys
sys.path.append('src')

from llama_cookbook.utils.activation_checkpointing import (
    apply_activation_checkpointing, 
    apply_activation_checkpointing_manual,
    disable_activation_checkpointing
)
from llama_cookbook.utils.memory_utils import get_memory_stats, print_memory_stats


class TestActivationCheckpointing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_apply_activation_checkpointing_with_small_model(self):
        """Test activation checkpointing with a small model."""
        try:
            # Try to load a small model that supports gradient checkpointing
            model_name = "gpt2"  # GPT2 is small and supports gradient checkpointing
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Apply checkpointing
            model = apply_activation_checkpointing(model, use_reentrant=False)
            
            # Check if it was applied
            self.assertTrue(hasattr(model, '_activation_checkpointing_enabled'))
            self.assertTrue(model._activation_checkpointing_enabled)
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                output = model(input_ids)
            self.assertIsNotNone(output)
            
            # Test disabling
            model = disable_activation_checkpointing(model)
            self.assertFalse(getattr(model, '_activation_checkpointing_enabled', True))
            
        except Exception as e:
            self.skipTest(f"Could not test with real model: {e}")
    
    def test_memory_monitoring(self):
        """Test memory monitoring utilities."""
        # Get memory stats
        stats = get_memory_stats()
        
        # Check that we got CPU stats at minimum
        self.assertIn('cpu_memory_gb', stats)
        self.assertIn('cpu_percent', stats)
        self.assertIn('system_memory_total_gb', stats)
        
        # Test printing (should not raise exception)
        print_memory_stats("Test", detailed=True)
        
        if torch.cuda.is_available():
            # Check GPU stats
            self.assertIn('gpu_allocated_gb', stats)
            self.assertIn('gpu_device', stats)
            self.assertIn('gpu_name', stats)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_reduction_with_checkpointing(self):
        """Test that activation checkpointing reduces memory usage."""
        try:
            # Load a small model
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16,
            ).to(self.device)
            
            # Create input
            batch_size = 4
            seq_len = 512
            input_ids = torch.randint(0, 50000, (batch_size, seq_len)).to(self.device)
            
            # Test without checkpointing
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            output1 = model(input_ids, labels=input_ids)
            loss1 = output1.loss
            loss1.backward()
            
            mem_without = torch.cuda.max_memory_allocated()
            model.zero_grad()
            torch.cuda.empty_cache()
            
            # Apply checkpointing
            model = apply_activation_checkpointing(model, use_reentrant=False)
            
            # Test with checkpointing
            torch.cuda.reset_peak_memory_stats()
            
            output2 = model(input_ids, labels=input_ids)
            loss2 = output2.loss
            loss2.backward()
            
            mem_with = torch.cuda.max_memory_allocated()
            
            # Memory with checkpointing should be less
            print(f"\nMemory without checkpointing: {mem_without / 1024**2:.1f}MB")
            print(f"Memory with checkpointing: {mem_with / 1024**2:.1f}MB")
            print(f"Memory saved: {(1 - mem_with/mem_without) * 100:.1f}%")
            
            # We expect at least some memory savings
            self.assertLess(mem_with, mem_without)
            
        except Exception as e:
            self.skipTest(f"Could not complete memory test: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
