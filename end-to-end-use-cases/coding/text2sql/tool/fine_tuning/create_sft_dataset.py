import argparse
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import sqlparse
from datasets import Dataset

from tqdm import tqdm


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": sample["messages"][0]["content"]},
            {"role": "user", "content": sample["messages"][1]["content"]},
            {"role": "assistant", "content": sample["messages"][2]["content"]},
        ]
    }


def create_sft_dataset(input_json, db_root_path):
    ds = []
    SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."

    for i, item in tqdm(enumerate(input_json)):
        print(f"processing #{i+1}")
        db_id = item["db_id"]
        question = item["question"]
        external_knowledge = item["evidence"]
        SQL = item["SQL"]
        db_path = db_root_path + "/" + item["db_id"] + "/" + item["db_id"] + ".sqlite"
        print(f"{db_path=}")
        prompt = generate_combined_prompts_one(
            db_path,
            question,
            knowledge=external_knowledge,
        )

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": SQL},
            ]
        }

        ds.append(example)

    dataset_dict = {key: [d[key] for d in ds] for key in ds[0]}
    dataset = Dataset.from_dict(dataset_dict)
    # dataset.save_to_disk(f"text2sql_sft_dataset")

    dataset = dataset.map(
        create_conversation, remove_columns=dataset.features, batched=False
    )
    dataset = dataset.train_test_split(test_size=0.3)

    dataset["train"].to_json("train_text2sql_sft_dataset.json", orient="records")
    dataset["test"].to_json("test_text2sql_sft_dataset.json", orient="records")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--input_json", type=str, required=True)
    args_parser.add_argument("--db_root_path", type=str, required=True)
    args = args_parser.parse_args()

    input_json = json.load(open(args.input_json, "r"))
    db_root_path = args.db_root_path

    create_sft_dataset(input_json, db_root_path)

# python create_sft_dataset.py --input_json ../data/train/train.json --db_root_path ../data/train/train_databases
