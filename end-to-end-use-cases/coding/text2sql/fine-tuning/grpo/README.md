# GRPO Fine-tuning for Text2SQL

This folder contains scripts to reinforcemen fine-tuning Llama models for the Text2SQL task using GRPO.

## Quick start

1. Download the BIRD train and dev datasets, if you haven't already:

```
git clone https://github.com/meta-llama/llama-cookbook
git checkout text2sql
cd llama-cookbook/end-to-end-use-cases/coding/text2sql/data
sh download_dev_unzip.sh
sh download_train_unzip.sh
```

2. (Optional) Set the following environment variable, so the reward of using LLM as a judge (via Llama 3.3 70b hosted on Together.ai) can be calculated:

```
pip install together
export TOGETHER_API_KEY=<your together.ai api key>
```

If you don't want to use the using LLM as a judge reward, you can comment out this [line](https://github.com/meta-llama/llama-cookbook/blob/text2sql/end-to-end-use-cases/coding/text2sql/fine-tuning/grpo/grpo_text2sql.py#L594) when calling GRPOTrainer and change the reward weights [here](https://github.com/meta-llama/llama-cookbook/blob/text2sql/end-to-end-use-cases/coding/text2sql/fine-tuning/grpo/grpo-llama323b-text2sql.yaml#L32) to [1.0, 3.0, 1.0]

3. Install the required libraries in a conda or virtual environment:

```
cd ../fine-tuning/grpo
pip install -r requirements.txt
```

4. Run the training script, assuming you have 6 GPUs to use for the training (if not, modify the `--num_processes` and `--gpu_ids`):

```
accelerate launch --num_processes 6 --gpu_ids 2,3,4,5,6,7 --config_file deepspeed_zero3.yaml grpo_text2sql.py --config grpo-llama323b-text2sql.yaml
```

You can modify the grpo-llama323b-text2sql.yaml file and tune `num_generations`, `learning_rate`, `reward_weights` and other parameters.

5. To evaluate a saved checkpoint, follow the steps [here](https://github.com/meta-llama/llama-cookbook/tree/text2sql/end-to-end-use-cases/coding/text2sql/eval#evaluation-with-llama-models-on-hugging-face-or-fine-tuned).
