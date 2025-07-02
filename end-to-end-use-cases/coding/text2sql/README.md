# Text2SQL: Evaluating and Fine-tuning Llama Models

This folder contains scripts to:

1. Evaluate Llama (original and fine-tuned) models on the Text2SQL task using the popular [BIRD](https://bird-bench.github.io) dataset in **3 simple steps**;

2. Generate fine-tuning datasets (both with and without CoT reasoning) and fine-tuning Llama 3.1 8B with the datasets, gaining a **165% (with no reasoning) and 209% (with reasoning) accuracy improvement** over the original model.

Our end goal is to maximize the accuracy of Llama models on the Text2SQL task. To do so we need to first evaluate the current state of the art Llama models on the task, then apply fine-tuning, agent and other approaches to evaluate and improve Llama's performance.

## Structure:

- data: contains the scripts to download the BIRD TRAIN and DEV datasets;
- eval: contains the scripts to evaluate Llama models (original and fine-tuned) on the BIRD dataset;
- fine-tune: contains the scripts to generate non-CoT and CoT datasets based on the BIRD TRAIN set and to fine-tune Llama models using the datasets;
- quickstart: contains a notebook to ask Llama 3.3 to convert natural language queries into SQL queries.

## Next Steps

1. Try GRPO RFT to further improve the accuracy.
2. Fine-tune Llama 3.3 70b and Llama 4 models.
3. Use torchtune.
4. Try agentic workflow.
5. Expand the eval to support other enterprise databases.
