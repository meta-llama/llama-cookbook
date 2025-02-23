## WIP

The end goal for this effort is to serve as fine-tuning data preparation kit.

## Current status:

Currently, I'm (WIP) evaluating the idea to improve tool-calling datasets. 

Setup:
- configs: Has the config prompts for creating synthetic data using `3.3`
- data_prep/scripts: This is what you would like to run to prepare your datasets for annotation
- scripts/annotation-inference: Script for generating synthetic datasets -> Use the vllm script for inference
- fine-tuning: configs for FT using TorchTune