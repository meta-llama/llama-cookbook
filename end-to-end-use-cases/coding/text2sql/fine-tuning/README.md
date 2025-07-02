# Llama Text2SQL Fine-tuning

This folder contains the scripts to generate datasets from the BIRD TRAIN set with and without CoT, and to supervised fine-tune (SFT), as the first step, the Llama 3.1 8B model: accuracy improvement of **165% on the fine-tuned model with no reasoning and 209% with reasoning** over the original model.

## SFT with the BIRD TRAIN dataset (No Reasoning)

We'll first use the BIRD TRAIN dataset to prepare for supervised fine-tuning with no reasoning info in the dataset.

### Using the TRAIN to prepare for supervised fine-tuning

1. Get the TRAIN dataset:
```
cd data
sh download_train_unzip.sh
```

2. Create the dataset

```
cd ../fine_tuning
python create_sft_dataset.py --input_json ../data/train/train.json --db_root_path ../data/train/train_databases
```

This will create `train_text2sql_sft_dataset.json` and `test_text2sql_sft_dataset.json` using the TRAIN set. Each line in the json files is in the conversation format ready for fine-tuning:

```
{"messages":[{"content":"You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement.","role":"system"},{"content":"-- DB Schema: <DB_SCHEMA>\n\n-- External Knowledge: <KNOWLEDGE_FROM_TRAIN>\n\n-- Question: <TEXT_QUESTION>","role":"user"},{"content":"<GOLD_SQL>","role":"assistant"}]}
```

### SFT (No Reasoning)

First, you need to login to HuggingFace (via running `huggingface-cli login` and enter your [HF token](https://huggingface.co/settings/tokens)) and have been granted access to the [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model.

Then run `python trl_sft.py`. After the fine-tuning completes, you'll see the fine-tuned model saved to `llama31-8b-text2sql-fine-tuned`, specified in `output_dir="llama31-8b-text2sql-fine-tuned"` of `TrainingArguments` in `trl_sft.py`.

After running `tensorboard --logdir ./llama31-8b-text2sql-fine_tuning` you can open `http://localhost:6006` to see the train loss chat etc:

![](fine_tuning/train_loss.png)


### Evaluating the fine-tuned model (No Reasoning)

First, modify `llama_eval.sh` to use the fine-tuned model:

```
YOUR_API_KEY='finetuned'
model='fine_tuning/llama31-8b-text2sql'
```

Then run `sh llama_eval.sh` to evaluate the fine-tuned model. The accuracy on the BIRD DEV dataset is about 37.16%. This is a 165% improvement over the model before fine-tuning, which has an accuracy of about 14.02% on the same dataset - you can confirm this by comparing the fine-tuned model's accuracy above with the original model's accuracy by modifying `llama_eval.sh` to use the original model:

```
YOUR_API_KEY='huggingface'
model='meta-llama/Llama-3.1-8B-Instruct'
```

Then running `sh llama_eval.sh` to evaluate the original model.

*Note:* We are using the 4-bit quantized Llama 3.1 8b model to reduce the memory footprint and improve the efficiency (as shown in the code nippet of llama_text2sql.py below), hence the accuracy of the quantized version (14.02%) is quite lower than the accuracy of the original Llama 3.1 8b (35.66%).

```
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
```

## SFT with the BIRD TRAIN dataset (With Reasoning)

Next we'll use the BIRD TRAIN dataset to prepare for supervised fine-tuning with reasoning info in the dataset. The goal is to see if we can improve the accuracy of the fine-tuned model by adding the reasoning info in the dataset.

### Creating a reasoning dataset from the TRAIN dataset

The script `create_reasoning_dataset.py` is used to create a reasoning dataset from the TRAIN dataset by asking Llama 3.3 70B to generate the reasoning for each text question and its corresponding gold SQL. The intent is to use the reasoning dataset to fine-tune the Llama model to improve the accuracy of the generated SQL.

To run the script, use the following commands:
```
python create_reasoning_dataset.py --input_json ../data/train/train.json --db_root_path ../data/train/train_databases
```

This will create a `text2sql_cot_dataset` dataset and `train_text2sql_cot_dataset.json` in the conversation format ready for fine-tuning. Each example in the dataset is generated from the code snippet below:

```
prompt = f"""
-- DB Schema: {db_schema}
-- External Knowledge: {external_knowledge}
-- Text Question: {question}
"""
cot = {
    "messages": [
        {
            "role": "system",
            "content": "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, generate the step-by-step reasoning and the final SQLite SQL select statement from the text question.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reasoning},
    ]
}
```

The prompt for Llama 3.3 70B to generate the `reasoning` above is:
```
You are a text to SQL query translator. Based on the DB Schema and External Knowledge, given the Text Question Input and its Gold SQL Output below, generate the step-by-step reasoning to infer the Gold SQL Output from the Text Question Input.

-- DB Schema: {db_schema}
-- External Knowledge: {external_knowledge}
-- Text Question Input: {question}
-- Gold SQL Output: {gold_SQL}

Your response should be as follows:\n\n
Let me think through this step by step:\n\n1. First, I need to consider...\n2. Then...\n3. Next...\n...\n\nFinally, the SQL statement for the text question is:
```sql ...```\n

"""
```

### SFT (With Reasoning)

Uncomment the line `# FT_DATASET = "train_text2sql_cot_dataset.json"` in trl_sft.py to use the reasoning dataset for fine-tuning. Then run `python trl_sft.py`. After the fine-tuning completes, you'll see the fine-tuned model saved to `llama31-8b-text2sql-fine-tuned`, specified in `output_dir="llama31-8b-text2sql-fine-tuned"` of `TrainingArguments` in `trl_sft.py` - you may want to rename the `output_dir` folder to something else to avoid overwriting the previous fine-tuned model.

The train loss chart will look like this:
![](fine_tuning/train_loss_cot.png)

### Evaluating the fine-tuned model (With Reasoning)

First, modify `llama_eval.sh` to use the fine-tuned model, which should match the `output_dir` in `TrainingArguments` in `trl_sft.py`:

```
YOUR_API_KEY='finetuned'
model='fine_tuning/llama31-8b-text2sql-fine-tuned'
```

Then uncomment the line `SYSTEM_PROMPT` [here](https://github.com/meta-llama/llama-cookbook/blob/text2sql/end-to-end-use-cases/coding/text2sql/eval/llama_text2sql.py#L31) in `llama_text2sql.py` to use it with the reasoning dataset fine-tuned model.

Now run `sh llama_eval.sh`, which will take longer because the reasoning is needed to generate the SQL. The accuracy this time is 43.37%, compared with 37.16% without reasoning. This is another 16% improvement over the model with fine-tuning without reasoning.
