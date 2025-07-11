# Llama Text2SQL Evaluation

We have updated and simplified the original eval scripts from the BIRD [repo](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) to 3 simple steps for Llama 3 & 4 models hosted via Meta's [Llama API](https://llama.developer.meta.com) or [Together.ai](https://together.ai), as well as the fine-tuned Llama 3.1 model.

## Evaluation Results

Below are the results of the Llama models we have evaluated on the BIRD DEV dataset:

| Model                  | Llama API Accuracy | Together Accuracy |
|------------------------|--------------------|-------------------|
| Llama 3.1 8b           | -                  | 35.66%            |
| Llama 3.3 70b          | 54.11%             | 54.63%            |
| Llama-3.1-405B         | -                  | 55.80%            |
| Llama 4 Scout          | 44.39%             | 43.94%            |
| Llama 4 Maverick       | 44.00%             | 41.46%            |

- Llama 3.1 8b on Hugging Face: quantized 14.02%, non-quantized 39.47%
- Fine-tuned with no CoT dataset: 39.31%
- Fine-tuned with CoT dataset: 43.35%

## Quick Start

First, run the commands below to create a new Conda environment and install all the required packages for Text2SQL evaluation and fine-tuning:

```
git clone https://github.com/meta-llama/llama-cookbook
cd llama-cookbook/end-to-end-use-cases/coding/text2sql
conda create -n llama-text2sql python=3.10
conda activate llama-text2sql
pip install -r requirements.txt
```

Then, follow the steps below to evaluate Llama 3 & 4 models on Text2SQL using the BIRD benchmark:

1. Get the DEV dataset:
```
cd data
sh download_dev_unzip.sh
cd ../eval
```

2. Open `llama_eval.sh` and set `YOUR_API_KEY` to your [Llama API](https://llama.developer.meta.com/) key or [Together](https://api.together.ai/) API key, then uncomment a line that starts with `model=` to specify the Llama model to use for the text2sql eval.

3. Run the evaluation script `sh llama_eval.sh`, which will use the BIRD DEV dataset (1534 examples in total) with external knowledge turned on to run the Llama model on each text question and compare the generated SQL with the gold SQL.

If your API key or model name is incorrect, the script will exit with an authentication or model not supported error.

After the script completes, you'll see the accuracy of the Llama model on the BIRD DEV text2sql. For example, the total accuracy is about 54.24% with `YOUR_API_KEY` set to your Llama API key and `model='Llama-3.3-70B-Instruct'`, or about 35.07% with `YOUR_API_KEY` set to your Together API key and `model=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`.

To compare your evaluated accuracy of your selected Llama model with other results in the BIRD Dev leaderboard, click [here](https://bird-bench.github.io/).

## Evaluation Process

1. **SQL Generation**: `llama_text2sql.py` sends natural language questions to the specified Llama model and collects the generated SQL queries.

2. **SQL Execution**: `text2sql_eval.py` executes both the generated SQL and ground truth SQL against the corresponding databases, then continues with steps 3 and 4 below.

3. **Result Comparison**: The results from executing the generated SQL are compared ([source code](text2sql_eval.py#L30)) with the results from the ground truth SQL to determine correctness.

4. **Accuracy Calculation**: Accuracy scores are calculated overall and broken down by difficulty levels (simple, moderate, challenging).

## Supported Models for Evaluation

Llama models supported on Together AI:
- meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
- meta-llama/Llama-3.3-70B-Instruct-Turbo
- meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- other Llama models hosted on Together AI

Llama models supported on Llama API:
- Llama-3.3-8B-Instruct
- Llama-3.3-70B-Instruct
- Llama-4-Maverick-17B-128E-Instruct-FP8
- Llama-4-Scout-17B-16E-Instruct-FP8
- other Llama models hosted on Llama API
