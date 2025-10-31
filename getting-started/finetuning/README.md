# Finetuning Llama


This folder contains instructions to fine-tune Meta Llama 3 on a

* [single-GPU setup](./singlegpu_finetuning.md)
* [multi-GPU setup](./multigpu_finetuning.md)

using the canonical [finetuning script](../../src/llama_cookbook/finetuning.py) in the llama-cookbook package.

If you are new to fine-tuning techniques, check out [an overview](./LLM_finetuning_overview.md).

> [!TIP]
> If you want to try finetuning Meta Llama 3 in a Jupyter notebook you can find a quickstart notebook [here](./quickstart_peft_finetuning.ipynb)

> [!NOTE]
> **New: Vision Fine-tuning Recipe** - Looking to fine-tune Llama 3.2 11B Vision for structured data extraction? Check out our comprehensive [vision fine-tuning recipe](./vision/README.md) that achieves 98% accuracy on document understanding tasks using torchtune, with detailed benchmarking comparing LoRA vs Full Parameter Fine-Tuning approaches.


## How to configure finetuning settings?

> [!TIP]
> All the setting defined in [config files](../../src/llama_cookbook/configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.


* [Training config file](../../src/llama_cookbook/configs/training.py) is the main config file that helps to specify the settings for our run and can be found in [configs folder](../../src/llama_cookbook/configs/)

It lets us specify the training settings for everything from `model_name` to `dataset_name`, `batch_size` and so on. Below is the list of supported settings:

```python
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False # shards model parameters, optimizer states and gradients across DDP ranks
    low_cpu_fsdp: bool=False # saves cpu memory by loading pretrained model on rank0 only
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85 # multiplicatively decay the learning rate by gamma after each epoch
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False # use parameter efficient fine tuning
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    freeze_LLM_only: bool = False # Freeze self-attention layers in the language_model. Vision model, multi_modal_projector, cross-attention will be fine-tuned
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler

```

* [Datasets config file](../../src/llama_cookbook/configs/datasets.py) provides the available options for datasets.

* [peft config file](../../src/llama_cookbook/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified. We currently support LoRA and Llama-Adapter. Please note that LoRA is the only technique which is supported in combination with FSDP.

* [FSDP config file](../../src/llama_cookbook/configs/fsdp.py) provides FSDP settings such as:

    * `mixed_precision` boolean flag to specify using mixed precision, defatults to true.

    * `use_fp16` boolean flag to specify using FP16 for mixed precision, defatults to False. We recommend not setting this flag, and only set `mixed_precision` that will use `BF16`, this will help with speed and memory savings while avoiding challenges of scaler accuracies with `FP16`.

    *  `sharding_strategy` this specifies the sharding strategy for FSDP, it can be:
        * `FULL_SHARD` that shards model parameters, gradients and optimizer states, results in the most memory savings.

        * `SHARD_GRAD_OP` that shards gradinets and optimizer states and keeps the parameters after the first `all_gather`. This reduces communication overhead specially if you are using slower networks more specifically beneficial on multi-node cases. This comes with the trade off of higher memory consumption.

        * `NO_SHARD` this is equivalent to DDP, does not shard model parameters, gradinets or optimizer states. It keeps the full parameter after the first `all_gather`.

        * `HYBRID_SHARD` available on PyTorch Nightlies. It does FSDP within a node and DDP between nodes. It's for multi-node cases and helpful for slower networks, given your model will fit into one node.

* `checkpoint_type` specifies the state dict checkpoint type for saving the model. `FULL_STATE_DICT` streams state_dict of each model shard from a rank to CPU and assembels the full state_dict on CPU. `SHARDED_STATE_DICT` saves one checkpoint per rank, and enables the re-loading the model in a different world size.

* `fsdp_activation_checkpointing` enables activation checkpoining for FSDP, this saves significant amount of memory with the trade off of recomputing itermediate activations during the backward pass. The saved memory can be re-invested in higher batch sizes to increase the throughput. We recommend you use this option.

* `pure_bf16` it moves the  model to `BFloat16` and if `optimizer` is set to `anyprecision` then optimizer states will be kept in `BFloat16` as well. You can use this option if necessary.


## Weights & Biases Experiment Tracking

You can enable [W&B](https://wandb.ai/) experiment tracking by using `use_wandb` flag as below. You can change the project name, entity and other `wandb.init` arguments in `wandb_config`.

```bash
python -m llama_cookbook.finetuning --use_peft --peft_method lora --quantization 8bit --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model --use_wandb
```
You'll be able to access a dedicated project or run link on [wandb.ai](https://wandb.ai) and see your dashboard like the one below.
<div style="display: flex;">
    <img src="../../../docs/img/wandb_screenshot.png" alt="wandb screenshot" width="500" />
</div>

## FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.
