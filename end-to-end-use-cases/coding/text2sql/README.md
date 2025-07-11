# Text2SQL: Evaluating and Fine-tuning Llama Models with CoT

This folder contains scripts to:

1. Evaluate Llama (original and fine-tuned) models on the Text2SQL task using the popular [BIRD](https://bird-bench.github.io) dataset in **3 simple steps**;

2. Generate two supervised fine-tuning (SFT) datasets (with and without CoT) and fine-tuning Llama 3.1 8B with the datasets, using different SFT options: with or without CoT, using quantization or not, full fine-tuning (FFT) or parameter-efficient fine-tuning (PEFT). The non-quantized PEFT SFT has the most performance gains: from 39.47% of the original Llama 3.1 8B model to 43.35%. (Note: the results are based on only 3 epochs of SFT.)

Our end goal is to maximize the accuracy of Llama models on the Text2SQL task. To do so we need to first evaluate the current state of the art Llama models on the task, then apply fine-tuning, agent and other approaches to evaluate and improve Llama's performance.

## Structure:

- data: contains the scripts to download the BIRD TRAIN and DEV datasets;
- eval: contains the scripts to evaluate Llama models (original and fine-tuned) on the BIRD dataset;
- fine-tune: contains the scripts to generate non-CoT and CoT datasets based on the BIRD TRAIN set and to fine-tune Llama models using the datasets;
- quickstart: contains a notebook to ask Llama 3.3 to convert natural language queries into SQL queries.

## Next Steps

1. Hyper-parameter tuning of the current SFT scripts.
2. Try GRPO RFT to further improve the accuracy.
3. Fine-tune Llama 3.3 70b and Llama 4 models.
4. Use torchtune.
5. Try agentic workflow.
6. Expand the eval to support other enterprise databases.
