"""
Memory monitoring utilities for training.
"""
import torch
import gc
import psutil
import os
from typing import Dict, Optional


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    stats = {}
    
    # GPU memory stats
    if torch.cuda.is_available():
        # Use current device instead of assuming device 0
        device = torch.cuda.current_device()
        
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(device) / 1024**3
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved(device) / 1024**3
        stats['gpu_free_gb'] = (torch.cuda.get_device_properties(device).total_memory - 
                                torch.cuda.memory_reserved(device)) / 1024**3
        stats['gpu_total_gb'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
        stats['gpu_device'] = device
        stats['gpu_name'] = torch.cuda.get_device_name(device)
    
    # CPU memory stats
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    stats['cpu_memory_gb'] = memory_info.rss / 1024**3
    stats['cpu_percent'] = process.memory_percent()
    
    # System-wide memory stats
    virtual_memory = psutil.virtual_memory()
    stats['system_memory_total_gb'] = virtual_memory.total / 1024**3
    stats['system_memory_available_gb'] = virtual_memory.available / 1024**3
    stats['system_memory_percent'] = virtual_memory.percent
    
    return stats


def print_memory_stats(stage: str = "", detailed: bool = False):
    """Print current memory usage statistics."""
    stats = get_memory_stats()
    
    if stage:
        print(f"\n[{stage}]")
    
    if 'gpu_allocated_gb' in stats:
        print(f"GPU Memory ({stats.get('gpu_name', 'Unknown')} - Device {stats.get('gpu_device', 0)}): "
              f"{stats['gpu_allocated_gb']:.2f}GB allocated, "
              f"{stats['gpu_reserved_gb']:.2f}GB reserved, "
              f"{stats['gpu_free_gb']:.2f}GB free")
        
        if detailed:
            utilization = (stats['gpu_allocated_gb'] / stats['gpu_total_gb']) * 100
            print(f"  GPU Utilization: {utilization:.1f}% of {stats['gpu_total_gb']:.2f}GB total")
    else:
        print("No GPU available")
    
    if detailed:
        print(f"Process Memory: {stats['cpu_memory_gb']:.2f}GB ({stats['cpu_percent']:.1f}% of system)")
        print(f"System Memory: {stats['system_memory_available_gb']:.2f}GB available "
              f"of {stats['system_memory_total_gb']:.2f}GB total "
              f"({stats['system_memory_percent']:.1f}% used)")


def clear_memory():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_peak_memory_stats() -> Dict[str, float]:
    """Get peak memory statistics since last reset."""
    stats = {}
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        stats['gpu_peak_allocated_gb'] = torch.cuda.max_memory_allocated(device) / 1024**3
        stats['gpu_peak_reserved_gb'] = torch.cuda.max_memory_reserved(device) / 1024**3
    
    return stats


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def track_memory_usage(func):
    """Decorator to track memory usage of a function."""
    def wrapper(*args, **kwargs):
        # Clear memory and get initial stats
        clear_memory()
        reset_peak_memory_stats()
        initial_stats = get_memory_stats()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get final stats
        final_stats = get_memory_stats()
        peak_stats = get_peak_memory_stats()
        
        # Calculate differences
        if 'gpu_allocated_gb' in initial_stats:
            gpu_diff = final_stats['gpu_allocated_gb'] - initial_stats['gpu_allocated_gb']
            peak_allocated = peak_stats.get('gpu_peak_allocated_gb', 0)
            print(f"\n[{func.__name__}] GPU Memory Impact:")
            print(f"  Current change: {gpu_diff:+.2f}GB")
            print(f"  Peak allocated: {peak_allocated:.2f}GB")
        
        cpu_diff = final_stats['cpu_memory_gb'] - initial_stats['cpu_memory_gb']
        print(f"  CPU memory change: {cpu_diff:+.2f}GB")
        
        return result
    
    return wrapper


class MemoryTrace:
    """Context manager for tracking memory usage during a code block."""
    def __init__(self, name: str = ""):
        self.name = name
        self.initial_stats = None
        
    def __enter__(self):
        clear_memory()
        reset_peak_memory_stats() if torch.cuda.is_available() else None
        self.initial_stats = get_memory_stats()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        final_stats = get_memory_stats()
        peak_stats = get_peak_memory_stats() if torch.cuda.is_available() else {}
        
        # Print memory usage report
        print(f"\n[MemoryTrace: {self.name if self.name else 'Block'}]")
        
        if 'gpu_allocated_gb' in self.initial_stats:
            initial_gpu = self.initial_stats['gpu_allocated_gb']
            final_gpu = final_stats['gpu_allocated_gb']
            gpu_diff = final_gpu - initial_gpu
            
            print(f"  GPU Memory Change: {gpu_diff:+.2f}GB "
                  f"({initial_gpu:.2f}GB → {final_gpu:.2f}GB)")
            
            if 'gpu_peak_allocated_gb' in peak_stats:
                peak_gpu = peak_stats['gpu_peak_allocated_gb']
                print(f"  GPU Peak Memory: {peak_gpu:.2f}GB")
        
        # CPU memory change
        initial_cpu = self.initial_stats['cpu_memory_gb']
        final_cpu = final_stats['cpu_memory_gb']
        cpu_diff = final_cpu - initial_cpu
        print(f"  CPU Memory Change: {cpu_diff:+.2f}GB "
              f"({initial_cpu:.2f}GB → {final_cpu:.2f}GB)")
        
        return False  # Don't suppress exceptions
