# Llama Text2SQL Evaluation

We have updated and simplified the original eval scripts from the BIRD [repo](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) to 3 simple steps for Llama 3 & 4 models hosted via Meta's [Llama API](https://llama.developer.meta.com), as well as Llama 3.1 8B on Hugging Face and its fine-tuned models.

## Evaluation Results

Below are the results of the Llama models we have evaluated on the BIRD DEV dataset:

| Model                  | Llama API Accuracy |
|------------------------|--------------------|
| Llama 3.1 8b           | 39.47% (*)         |
| Llama 3.3 70b          | 54.11%             |
| Llama 4 Scout          | 44.39%             |
| Llama 4 Maverick       | 44.00%             |

- Since Llama API does not have Llama 3.1 8b model, we use Hugging Face weights and vllm to run locally.

## Quick Start with Llama Models via Llama API

First, run the commands below to create a new Conda environment and install all the required packages for Text2SQL evaluation and fine-tuning:

```
conda create -n llama-text2sql python=3.10
conda activate llama-text2sql
git clone https://github.com/meta-llama/llama-cookbook
git checkout text2sql # to be removed after the PR merge
cd llama-cookbook/end-to-end-use-cases/coding/text2sql/eval
pip install -r requirements.txt
```

Then, follow the steps below to evaluate Llama 3 & 4 models on Text2SQL using the BIRD benchmark:

1. Get the DEV dataset:
```
cd ../data
sh download_dev_unzip.sh
cd ../eval
```

2. Open `llama_eval.sh` and set `YOUR_API_KEY` to your [Llama API](https://llama.developer.meta.com/) key then uncomment a line that starts with `model=` to specify the Llama model to use for the text2sql eval.

3. Run the evaluation script `sh llama_eval.sh`, which will use the BIRD DEV dataset (1534 examples in total) with external knowledge turned on to run the Llama model on each text question and compare the generated SQL with the gold SQL.

If your API key or model name is incorrect, the script will exit with an authentication or model not supported error.

After the script completes, you'll see the accuracy of the Llama model on the BIRD DEV text2sql. For example, the total accuracy is about 54.24% with `YOUR_API_KEY` set to your Llama API key and `model='Llama-3.3-70B-Instruct'`

To compare your evaluated accuracy of your selected Llama model with other results in the BIRD Dev leaderboard, click [here](https://bird-bench.github.io/).

## Evaluation with Llama Models on Hugging Face or Fine-tuned

We use vllm OpenAI compatible server to run Llama 3.1 8B on Hugging Face (steps below) or its fine-tuned models (steps [here](../fine-tuning/#evaluating-the-fine-tuned-model) for eval:

1. Uncomment the last two lines in requirements.txt then run `pip install -r requirements.txt`:
```
# vllm==0.9.2
# openai==1.90.0
```

2. Uncomment in `llama_eval.sh`:
```
YOUR_API_KEY='huggingface'
model='meta-llama/Llama-3.1-8B-Instruct'
```

3. Start the vllm server:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 8192 --max-num-seqs 64
```
or if you have multiple GPUs, do something like:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --max-num-batched-tokens 8192 --max-num-seqs 64
```

then run `sh llama_eval.sh`.

## Evaluation Process

1. **SQL Generation**: `llama_text2sql.py` sends natural language questions to the specified Llama model and collects the generated SQL queries.

2. **SQL Execution**: `text2sql_eval.py` executes both the generated SQL and ground truth SQL against the corresponding databases, then continues with steps 3 and 4 below.

3. **Result Comparison**: The results from executing the generated SQL are compared ([source code](text2sql_eval.py#L29)) with the results from the ground truth SQL to determine correctness.

4. **Accuracy Calculation**: Accuracy scores are calculated overall and broken down by difficulty levels (simple, moderate, challenging).
