import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
from typing import Dict

from llama_api_client import LlamaAPIClient
from tqdm import tqdm

MAX_NEW_TOKENS = 10240  # If API has max tokens (vs max new tokens), we calculate it
TIMEOUT = 60  # Timeout in seconds for each API call


def local_llama(client, api_key, prompts, model, max_workers=8):
    """
    Process multiple prompts in parallel using the vllm server.

    Args:
        client: OpenAI client
        prompts: List of prompts to process
        model: Model name
        max_workers: Maximum number of parallel workers

    Returns:
        List of results in the same order as prompts
    """

    SYSTEM_PROMPT = (
        (
            "You are a text to SQL query translator. Using the SQLite DB Schema "
            "and the External Knowledge, translate the following text question "
            "into a SQLite SQL select statement."
        )
        if api_key == "huggingface"
        else (
            "You are a text to SQL query translator. Using the SQLite DB Schema "
            "and the External Knowledge, generate the step-by-step reasoning and "
            "then the final SQLite SQL select statement from the text question."
        )
    )

    def process_single_prompt(prompt):
        messages = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"role": "user", "content": prompt},
        ]
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=TIMEOUT,
                temperature=0,
            )
            answer = chat_response.choices[0].message.content.strip()

            pattern = re.compile(r"```sql\n*(.*?)```", re.DOTALL)
            matches = pattern.findall(answer)
            if not matches:
                result = answer
            else:
                result = matches[0]

            return result
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return f"error:{e}"

    print(
        f"local_llama: Processing {len(prompts)} prompts with {model=} "
        f"using {max_workers} workers"
    )
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a map of futures to their indices
        future_to_index = {
            executor.submit(process_single_prompt, prompt): i
            for i, prompt in enumerate(prompts)
        }

        # Initialize results list with None values
        results = [None] * len(prompts)

        # Process completed futures as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(prompts),
            desc="Processing prompts",
        ):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error processing prompt at index {index}: {e}")
                results[index] = f"error:{e}"

    return results


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

    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )
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


def cloud_llama(client, api_key, model, prompts):
    """
    Process multiple prompts sequentially using the cloud API, showing progress with tqdm.

    Args:
        client: LlamaAPIClient
        api_key: API key
        model: Model name
        prompts: List of prompts to process (or a single prompt as string)

    Returns:
        List of results if prompts is a list, or a single result if prompts is a string
    """
    SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."

    # Handle the case where a single prompt is passed
    single_prompt = False
    if isinstance(prompts, str):
        prompts = [prompts]
        single_prompt = True

    results = []

    # Process each prompt sequentially with tqdm progress bar
    for prompt in tqdm(prompts, desc="Processing prompts", unit="prompt"):
        try:
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
        except Exception as e:
            result = "error:{}".format(e)
            print(f"{result=}")

        results.append(result)

    # Return a single result if input was a single prompt
    if single_prompt:
        return results[0]
    return results


def batch_collect_response_from_llama(
    db_path_list, question_list, api_key, model, knowledge_list=None, batch_size=8
):
    """
    Process multiple questions in parallel using the vllm server.

    Args:
        db_path_list: List of database paths
        question_list: List of questions
        api_key: API key
        model: Model name
        knowledge_list: List of knowledge strings (optional)
        batch_size: Number of parallel requests

    Returns:
        List of SQL responses
    """
    if api_key in ["huggingface", "finetuned"]:
        from openai import OpenAI

        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    else:
        client = LlamaAPIClient()

    # Generate all prompts first
    prompts = []
    for i, question in enumerate(question_list):
        if knowledge_list:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], question=question, knowledge=knowledge_list[i]
            )
        else:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], question=question
            )
        prompts.append(cur_prompt)

    print(f"Generated {len(prompts)} prompts for Llama processing")

    if api_key in [
        "huggingface",
        "finetuned",
    ]:
        # Process prompts in parallel; running vllm on multiple GPUs for best eval performance
        results = local_llama(
            client=client,
            api_key=api_key,
            prompts=prompts,
            model=model,
            max_workers=batch_size,
        )
    else:
        results = cloud_llama(
            client=client,
            api_key=api_key,
            model=model,
            prompts=prompts,
        )

    # Format results
    response_list = []
    for i, result in enumerate(results):
        if isinstance(result, str):
            sql = result
        else:
            sql = "SELECT" + result["choices"][0]["text"]

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
    args_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of parallel requests for batch processing",
    )
    args = args_parser.parse_args()

    if args.api_key not in ["huggingface", "finetuned"]:
        os.environ["LLAMA_API_KEY"] = args.api_key

        try:
            # test if the Llama API key is valid
            client = LlamaAPIClient()
            client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "125*125 is?"}],
                temperature=0,
            )
        except Exception as exception:
            print(f"{exception=}")
            exit(1)

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    if args.use_knowledge == "True":
        responses = batch_collect_response_from_llama(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            model=args.model,
            knowledge_list=knowledge_list,
            batch_size=args.batch_size,
        )
    else:
        responses = batch_collect_response_from_llama(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            model=args.model,
            knowledge_list=None,
            batch_size=args.batch_size,
        )

    output_name = args.data_output_path + "predict_" + args.mode + ".json"

    generate_sql_file(sql_lst=responses, output_path=output_name)

    print("successfully collect results from {}".format(args.model))
