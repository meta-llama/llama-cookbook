# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Import existing utilities
from llama_cookbook.utils.dataset_utils import *
from llama_cookbook.utils.fsdp_utils import fsdp_auto_wrap_policy, hsdp_device_mesh, get_policies
from llama_cookbook.utils.train_utils import *

# Import new activation checkpointing utilities
from llama_cookbook.utils.activation_checkpointing import (
    apply_activation_checkpointing,
    disable_activation_checkpointing
)

# Import new memory utilities
from llama_cookbook.utils.memory_utils import (
    MemoryTrace,
    get_memory_stats,
    print_memory_stats,
    clear_memory,
    track_memory_usage,
    get_peak_memory_stats,
    reset_peak_memory_stats
)
