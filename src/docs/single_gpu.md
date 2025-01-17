# Fine-tuning with Single GPU

To run fine-tuning on a single GPU, we will  make use of two packages

1- [PEFT](https://huggingface.co/blog/peft) methods and in specific using HuggingFace [PEFT](https://github.com/huggingface/peft)library.

2- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) int8 quantization.

Given combination of PEFT and Int8 quantization, we would be able to fine_tune a Meta Llama 3 8B model on one consumer grade GPU such as A10.

## Requirements
To run the examples, make sure to install the llama-cookbook package (See [README.md](../README.md) for details).

**Please note that the llama-cookbook package will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

## How to run it?

Get access to a machine with one GPU or if using a multi-GPU machine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id` and run the following. It runs by default with `samsum_dataset` for summarization application.

**NOTE** To run the fine-tuning with `QLORA`, make sure to set `--peft_method lora` and `--quantization int4`.

```bash

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization 8bit --use_fp16 --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model

```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`.

* `--quantization` boolean flag to enable int8 quantization


## How to run with different datasets?

Currently 4 datasets are supported that can be found in [Datasets config file](../llama_recipes/configs/datasets.py).

* `grammar_dataset` : use this [notebook](../llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process theJfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `ft_dataset` folder.

```bash
wget -P src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization 8bit --dataset grammar_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization 8bit --dataset alpaca_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization 8bit --dataset samsum_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model

```

## Where to configure settings?

* [Training config file](../llama_recipes/configs/training.py) is the main config file that help to specify the settings for our run can be found in

It let us specify the training settings, everything from `model_name` to `dataset_name`, `batch_size` etc. can be set here. Below is the list of supported settings:

```python
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
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
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
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

* [Datasets config file](../llama_recipes/configs/datasets.py)
    ../src/llama_recipes/configs/datasets.py) provides the available options for datasets.

* [peft config file](../llama_recipes/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified.

## FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.
