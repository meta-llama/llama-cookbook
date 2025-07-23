import argparse
import json
import os
import re
import sqlite3

from datasets import Dataset, load_from_disk
from langchain_together import ChatTogether
from llama_api_client import LlamaAPIClient

if (
    os.environ.get("LLAMA_API_KEY", "") == ""
    and os.environ.get("TOGETHER_API_KEY", "") == ""
):
    print(
        "Please set the environment variable LLAMA_API_KEY or TOGETHER_API_KEY to your API key."
    )
    exit(1)


if os.environ.get("LLAMA_API_KEY", "") != "":  # Llama model on Llama API
    try:
        client = LlamaAPIClient(api_key=os.environ["LLAMA_API_KEY"])

        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": "125*125 is?"}],
            temperature=0,
        )
        answer = response.completion_message.content.text
    except Exception as exception:
        print(f"Invalid LLAMA_API_KEY {exception=}")

if os.environ.get("TOGETHER_API_KEY", "") != "":  # Llama model on together
    llm = ChatTogether(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0,
    )
    try:
        answer = llm.invoke("125*125 is?").content
    except Exception as exception:
        print(f"Invalid TOGETHER_API_KEY - {exception=}")
        exit(1)


def llama(prompt, model="Llama-3.3-70B-Instruct"):

    if os.environ["LLAMA_API_KEY"] != "":
        client = LlamaAPIClient(api_key=os.environ["LLAMA_API_KEY"])
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response.completion_message.content.text
    else:
        llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0,
        )
        answer = llm.invoke(prompt).content
        return answer


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

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": sample["messages"][0]["content"]},
            {"role": "user", "content": sample["messages"][1]["content"]},
            {"role": "assistant", "content": sample["messages"][2]["content"]},
        ]
    }


def create_cot_dataset(input_json, db_root_path):
    cot_list = []
    diff = 0
    for i, item in enumerate(input_json):
        print(f"processing #{i+1}")

        db_id = item["db_id"]
        question = item["question"]
        external_knowledge = item["evidence"]
        gold_SQL = item["SQL"].strip()
        db_path = db_root_path + "/" + db_id + "/" + db_id + ".sqlite"
        # print(f"{db_path=}")
        db_schema = generate_schema_prompt(db_path)

        prompt_to_generate_reasoning = """
        You are a text to SQL query translator. Based on the DB Schema and External Knowledge, given the Text Question Input and its Gold SQL Output below, generate the step-by-step reasoning to infer the Gold SQL Output from the Text Question Input.

        -- DB Schema: {db_schema}
        -- External Knowledge: {external_knowledge}
        -- Text Question Input: {question}
        -- Gold SQL Output: {gold_SQL}

        Your response should be as follows:\n\n
        Let me think through this step by step:\n\n1. First, I need to consider...\n2. Then...\n3. Next...\n...\n\nFinally, the SQL statement for the text question is: 
        ```sql ...```\n

        """

        prompt_to_generate_reasoning = (
            prompt_to_generate_reasoning.replace("{db_schema}", db_schema)
            .replace("{external_knowledge}", external_knowledge)
            .replace("{question}", question)
            .replace("{gold_SQL}", gold_SQL)
        )
        reasoning = llama(prompt_to_generate_reasoning)

        pattern = re.compile(r"```sql\n*(.*?)```", re.DOTALL)
        matches = pattern.findall(reasoning)
        if matches != []:
            gene_SQL = matches[0].replace("\n", "").strip()
            gene_SQL = re.sub(r"\s{2,}", " ", gene_SQL)
        else:
            gene_SQL = reasoning

        print(f"{diff=}\n{gold_SQL=}\n{gene_SQL=}")
        if gold_SQL != gene_SQL:
            diff += 1
            continue

        # use the reasoning generated above to generate an example for the reasoning dataset used for fine-tuning
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
        cot_list.append(cot)

    print(f"{diff=}, total: {len(input_json)}")
    dataset_dict = {key: [d[key] for d in cot_list] for key in cot_list[0]}
    hf_dataset = Dataset.from_dict(dataset_dict)
    hf_dataset.save_to_disk("text2sql_cot_dataset")

    dataset = load_from_disk("text2sql_cot_dataset")
    dataset = dataset.map(
        create_conversation, remove_columns=dataset.features, batched=False
    )
    dataset = dataset.train_test_split(test_size=0.3)

    dataset["train"].to_json("train_text2sql_cot_dataset.json", orient="records")
    dataset["test"].to_json("test_text2sql_cot_dataset.json", orient="records")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--input_json", type=str, required=True)
    args_parser.add_argument("--db_root_path", type=str, required=True)
    args = args_parser.parse_args()

    input_json = json.load(open(args.input_json, "r"))
    db_root_path = args.db_root_path

    create_cot_dataset(input_json, db_root_path)

# python create_reasoning_dataset.py --input_json ../data/train/train.json --db_root_path ../data/train/train_databases
