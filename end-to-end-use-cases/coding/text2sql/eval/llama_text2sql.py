import argparse
import fnmatch
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd
import sqlparse

import torch
from datasets import Dataset, load_dataset
from langchain_together import ChatTogether
from llama_api_client import LlamaAPIClient
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

MAX_NEW_TOKENS=10240  # If API has max tokens (vs max new tokens), we calculate it

def local_llama(prompt, pipe):
    SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."
    # UNCOMMENT TO USE THE FINE_TUNED MODEL WITH REASONING DATASET
    # SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, generate the step-by-step reasoning and the final SQLite SQL select statement from the text question."

    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"role": "user", "content": prompt},
    ]

    raw_prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"local_llama: {raw_prompt=}")

    outputs = pipe(
        raw_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        top_k=50,
        top_p=0.1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )

    answer = outputs[0]["generated_text"][len(raw_prompt) :].strip()

    pattern = re.compile(r"```sql\n*(.*?)```", re.DOTALL)
    matches = pattern.findall(answer)
    if matches != []:
        result = matches[0]
    else:
        result = answer

    print(f"{result=}")
    return result


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    """
    Read an sqlite file, and return the CREATE commands for each of the tables in the database.
    """
    asdf = "database" if bench_root == "spider" else "databases"
    with sqlite3.connect(
        f"file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro", uri=True
    ) as conn:
        # conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                    table[0]
                )
            )
            schemas[table[0]] = cursor.fetchone()[0]

        return schemas


def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [
        max(len(str(value[i])) for value in values + [column_names])
        for i in range(len(column_names))
    ]

    # Print the column names
    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )
    # print(header)
    # Print the values
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + "\n" + rows
    return final_output


def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    """
    :param root_place:
    :param db_name:
    :return:
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == "sqlite_sequence":
            continue
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                table[0]
            )
        )
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ["order", "by", "group"]:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                num_rows, cur_table, num_rows, rows_prompt
            )
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "-- DB Schema: " + "\n\n".join(full_schema_prompt_list)

    return schema_prompt


def generate_comment_prompt(question, knowledge=None):
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)
    question_prompt = "-- Question: {}".format(question)

    result_prompt = knowledge_prompt + "\n\n" + question_prompt

    return result_prompt


def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + "\n\n" + comment_prompt

    return combined_prompts


def cloud_llama(api_key, model, prompt, max_tokens, temperature, stop):

    SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."
    try:
        if model.startswith("meta-llama/"):
            final_prompt = SYSTEM_PROMPT + "\n\n" + prompt
            final_max_tokens = len(final_prompt) + MAX_NEW_TOKENS
            llm = ChatTogether(
                model=model,
                temperature=0,
                max_tokens=final_max_tokens,
            )
            answer = llm.invoke(final_prompt).content
        else:
            client = LlamaAPIClient()
            messages = [
                {"content": SYSTEM_PROMPT, "role": "system"},
                {"role": "user", "content": prompt},
            ]
            final_max_tokens = len(messages) + MAX_NEW_TOKENS
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=final_max_tokens,
            )
            answer = response.completion_message.content.text

        pattern = re.compile(r"```sql\n*(.*?)```", re.DOTALL)
        matches = pattern.findall(answer)
        if matches != []:
            result = matches[0]
        else:
            result = answer

        print(result)
    except Exception as e:
        result = "error:{}".format(e)
        print(f"{result=}")
    return result


def huggingface_finetuned(api_key, model):
    if api_key == "finetuned":
        model_id = model

        # Check if this is a PEFT model by looking for adapter_config.json
        import os

        is_peft_model = os.path.exists(os.path.join(model_id, "adapter_config.json"))

        if is_peft_model:
            # Use AutoPeftModelForCausalLM for PEFT fine-tuned models
            print(f"Loading PEFT model from {model_id}")
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            # Use AutoModelForCausalLM for FFT (Full Fine-Tuning) models
            print(f"Loading FFT model from {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # For FFT models, handle pad token if it was added during training
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        tokenizer.padding_side = "right"  # to prevent warnings

    elif api_key == "huggingface":
        model_id = model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,  # None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def collect_response_from_llama(
    db_path_list, question_list, api_key, model, knowledge_list=None
):
    """
    :param db_path: str
    :param question_list: []
    :return: dict of responses
    """
    responses_dict = {}
    response_list = []

    if api_key in ["huggingface", "finetuned"]:
        pipe = huggingface_finetuned(api_key=api_key, model=model)

    for i, question in tqdm(enumerate(question_list)):
        print(
            "--------------------- processing question #{}---------------------".format(
                i + 1
            )
        )
        print("the question is: {}".format(question))

        if knowledge_list:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], question=question, knowledge=knowledge_list[i]
            )
        else:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], question=question
            )

        if api_key in ["huggingface", "finetuned"]:
            plain_result = local_llama(prompt=cur_prompt, pipe=pipe)
        else:
            plain_result = cloud_llama(
                api_key=api_key,
                model=model,
                prompt=cur_prompt,
                max_tokens=10240,
                temperature=0,
                stop=["--", "\n\n", ";", "#"],
            )
        if type(plain_result) == str:
            sql = plain_result
        else:
            sql = "SELECT" + plain_result["choices"][0]["text"]

        # responses_dict[i] = sql
        db_id = db_path_list[i].split("/")[-1].split(".sqlite")[0]
        sql = (
            sql + "\t----- bird -----\t" + db_id
        )  # to avoid unpredicted \t appearing in codex results
        response_list.append(sql)

    return response_list


def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data["question"])

    return question_list


def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data["evidence"])

    return knowledge_list


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])

    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="True")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument("--api_key", type=str, required=True)
    args_parser.add_argument("--model", type=str, required=True)
    args_parser.add_argument("--data_output_path", type=str)
    args = args_parser.parse_args()

    if not args.api_key in ["huggingface", "finetuned"]:
        if args.model.startswith("meta-llama/"):  # Llama model on together

            os.environ["TOGETHER_API_KEY"] = args.api_key
            llm = ChatTogether(
                model=args.model,
                temperature=0,
            )
            try:
                response = llm.invoke("125*125 is?").content
                print(f"{response=}")
            except Exception as exception:
                print(f"{exception=}")
                exit(1)
        else:  # Llama model on Llama API
            os.environ["LLAMA_API_KEY"] = args.api_key

            try:
                client = LlamaAPIClient()

                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": "125*125 is?"}],
                    temperature=0,
                )
                answer = response.completion_message.content.text

                print(f"{answer=}")
            except Exception as exception:
                print(f"{exception=}")
                exit(1)

    eval_data = json.load(open(args.eval_path, "r"))
    # '''for debug'''
    # eval_data = eval_data[:3]
    # '''for debug'''

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    if args.use_knowledge == "True":
        responses = collect_response_from_llama(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            model=args.model,
            knowledge_list=knowledge_list,
        )
    else:
        responses = collect_response_from_llama(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            model=args.model,
            knowledge_list=None,
        )

    output_name = args.data_output_path + "predict_" + args.mode + ".json"

    generate_sql_file(sql_lst=responses, output_path=output_name)

    print("successfully collect results from {}".format(args.model))
