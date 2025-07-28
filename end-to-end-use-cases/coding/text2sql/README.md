# Improving Llama Text2SQL performance with CoT Fine-tuning

This recipe is step by step guide to improve Llama performance on Text2SQL measured with the popular [BIRD](https://bird-bench.github.io) benchmark. We generate a synthetic Chain of Thought(CoT) dataset and fine-tune Llama models on it.

Results:
|-----------------------------|-------------------------------|
| baseline                    | 39.47%                        |
| CoT, PEFT                   | 43.35%                        |
| CoT, FFT                    | 42.44% (3 epochs)             |
| CoT, FFT                    | 43.87% (10 epochs)            |

The complete steps are:

1. Pre-processing the BIRD TRAIN datset by converting SQL statements into the conversation format.

2. We use the conversations from step 1, add CoT to these existing conversations using Llama-3.3-70B.

3. Fine-tuning Llama-3.1-8B on the dataset from step 2.

4. We provide scripts to simplify running the [BIRD](https://bird-bench.github.io) eval benchmark on the fine-tuned models and compare it with out of the model.

## Folder Structure

- quickstart folder: contains a notebook to ask Llama 3.3 to convert natural language queries into SQL queries.
- data folder: contains scripts to download the BIRD TRAIN and DEV datasets;
- fine-tune folder: contains scripts to generate non-CoT and CoT datasets based on the BIRD TRAIN set and to supervised fine-tune Llama models using the datasets, with different SFT options (quantization or not, full fine-tuning or parameter-efficient fine-tuning);
- eval folder: contains scripts to evaluate Llama models (original and fine-tuned) on the BIRD dataset;

We also experimented with supervised fine-tuning (SFT) without CoT which resulted in slightly lower accuracy.

## Next Steps

1. Hyper-parameter tuning of the current SFT scripts.
2. Try GRPO reinforcement learning to further improve the accuracy.
3. Fine-tune Llama 3.3 70B and Llama 4 models.
4. Try agentic workflow.
5. Expand the eval to support other enterprise databases.
