## WIP

### Instructions to run:

Time to setup: ~20-30 minutes:
- Grab your HF and Wandb.ai API key-you will need those
- Steps to install:

 ```
conda create -n test-ft python=3.10
conda activate test-ft
pip install --pre torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
pip install transformers datasets wandb
pip install huggingface-cli
huggingface-cli login
wandb login


git clone https://github.com/meta-llama/llama-cookbook/
cd llama-cookbook/
git checkout data-tool
cd end-to-end-use-cases/data-tool/scripts/finetuning
tune download meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir /tmp/Meta-Llama-3.1-70B-Instruct --ignore-patterns "original/consolidated*
tune run --nproc_per_node 8 full_finetune_distributed --config ft-config.yaml
 ```

The end goal for this effort is to serve as fine-tuning data preparation kit.

## Current status:

Currently, I'm (WIP) evaluating the idea to improve tool-calling datasets. 

Setup:
- configs: Has the config prompts for creating synthetic data using `3.3`
- data_prep/scripts: This is what you would like to run to prepare your datasets for annotation
- scripts/annotation-inference: Script for generating synthetic datasets -> Use the vllm script for inference
- fine-tuning: configs for FT using TorchTune