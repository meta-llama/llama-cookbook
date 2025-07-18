import argparse
import json
import os
import re
import sqlite3
from typing import Dict, List, Tuple

from tqdm import tqdm

from vllm import LLM, EngineArgs, SamplingParams

DEFAULT_MAX_TOKENS=10240
SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."
# UNCOMMENT TO USE THE FINE_TUNED MODEL WITH REASONING DATASET
# SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, generate the step-by-step reasoning and the final SQLite SQL select statement from the text question."


def inference(llm, sampling_params, user_prompt):
    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"role": "user", "content": user_prompt},
    ]

    print(f"{messages=}")

    response = llm.chat(messages, sampling_params, use_tqdm=False)
    print(f"{response=}")
    response_text = response[0].outputs[0].text
    pattern = re.compile(r"```sql\n*(.*?)```", re.DOTALL)
    matches = pattern.findall(response_text)
    if matches != []:
        result = matches[0]
    else:
        result = response_text
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



def collect_response_from_llama(
    llm, sampling_params, db_path_list, question_list, knowledge_list=None
):
    response_list = []

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

        plain_result = inference(llm, sampling_params, cur_prompt)
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
    args_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args_parser.add_argument("--temperature", type=float, default=0.0)
    args_parser.add_argument("--top_k", type=int, default=50)
    args_parser.add_argument("--top_p", type=float, default=0.1)
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path, "r"))
    # '''for debug'''
    # eval_data = eval_data[:3]
    # '''for debug'''

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    llm = LLM(model=args.model, download_dir="/opt/hpcaas/.mounts/fs-06ad2f76a5ad0b18f/shared/amiryo/.cache/vllm")
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = args.max_tokens
    sampling_params.temperature = args.temperature
    sampling_params.top_p = args.top_p
    sampling_params.top_k = args.top_k


    if args.use_knowledge == "True":
        responses = collect_response_from_llama(
            llm=llm,
            sampling_params=sampling_params,
            db_path_list=db_path_list,
            question_list=question_list,
            knowledge_list=knowledge_list,
        )
    else:
        responses = collect_response_from_llama(
            llm=llm,
            sampling_params=sampling_params,
            db_path_list=db_path_list,
            question_list=question_list,
            knowledge_list=None,
        )

    output_name = args.data_output_path + "predict_" + args.mode + ".json"

    generate_sql_file(sql_lst=responses, output_path=output_name)

    print("successfully collect results from {}".format(args.model))
