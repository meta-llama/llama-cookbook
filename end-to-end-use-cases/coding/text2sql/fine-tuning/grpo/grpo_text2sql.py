import json
import logging

import os
import random
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List

from datasets import Dataset
from func_timeout import func_timeout, FunctionTimedOut
from together import Together
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import get_peft_config, GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
TRAIN_JSON = "../../data/train/train.json"
DB_ROOT_PATH = "../../data/train/train_databases/"
LOG_REWARD_FILE_NAME = "text2sql_grpo_rewards5.log"
COMPLETION_SAMPLE_TXT_FILE_NAME = "completion_samples5.txt"


def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents


def execute_sql(predicted_sql, ground_truth_dbid):
    ground_truth, db_name = ground_truth_dbid.split("\t----- bird -----\t")

    # print(f"\n==== execute_sql ====\n{predicted_sql=}\n{ground_truth=}")

    db_path = DB_ROOT_PATH + db_name + "/" + db_name + ".sqlite"
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0

    if set(predicted_res) == set(ground_truth_res):
        res = 1
        print("execution result same")
    else:
        print("execution result different")
    conn.close()

    return res


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################


def log_reward(reason, completion, gt):
    import os

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", LOG_REWARD_FILE_NAME)
    with open(log_file, "a") as f:
        f.write("\n\n==============\n")
        f.write(f">>>{reason=}\n>>>{completion=}\n>>>{gt=}\n")


def extract_answer(text):
    """
    Extracts the final SQL statement answer from the raw text.
    """
    try:
        match = re.search(r"#### (\-?[\d\.,$]+)", text)
        if match:
            matched_string = match.group(1)

            # Remove any characters that would cause a ValueError,
            # such as dollar signs ($) and commas (,)
            cleaned_string = re.sub(r"[$,]", "", matched_string)

            return float(cleaned_string)

        match = re.search(
            r"(?:The final answer is|The answer is):?\s*(\-?[\d\.,$]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            matched_string = match.group(1)
            cleaned_string = re.sub(r"[$,]", "", matched_string)
            return float(cleaned_string)

    except (ValueError, AttributeError):
        print(f"Error extracting answer from text: {match.group(1)}")
        pass
    return None


def format_reward_func(completions, answer, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        answer (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, answer):

        try:
            if random.random() < 0.1:  # 1% chance to write samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join(
                    "completion_samples", COMPLETION_SAMPLE_TXT_FILE_NAME
                )
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)

            # Check if the format is correct
            regex = r"<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                log_reward("format_reward 0", completion, gt)
            else:
                rewards.append(1.0)
                log_reward("format_reward 1", completion, gt)
        except Exception as e:
            rewards.append(0.0)
            log_reward(f"format_reward 0 - exception {e=}", completion, gt)
    return rewards


def execution_reward_func(completions, answer, **kwargs):
    """
    Evaluates completions based on SQL statement execution result

    Args:
        completions (list[str]): Generated outputs
        answer (list[str]): Ground truth answers (from content with "role":"assistant" in train_text2sql_sft_dataset.json)

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        try:
            # gt = extract_answer(gt)
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                log_reward("execution_reward 0 - no answer tag found", completion, gt)
                continue
            # Extract the "answer" part from the completion
            predicted_sql = match.group(1).strip()

            reason = "execution result different"
            # execute the sql_generated and gt and compare the results
            try:
                res = func_timeout(
                    30.0,
                    execute_sql,
                    args=(predicted_sql, gt),
                )
            except KeyboardInterrupt:
                sys.exit(0)
            except FunctionTimedOut:
                print("FunctionTimedOut")
                reason = "execution timeout"
                res = 0
            except Exception as e:
                print("Exception", e)
                reason = f"execution exception {e}"
                res = 0

            if res == 1:
                # reason = "execution result same"
                rewards.append(1.0)
                log_reward("execution_reward 1", completion, gt)
            else:
                rewards.append(0.0)
                log_reward(
                    f"execution_reward 0 {reason=}, {predicted_sql=}",
                    completion,
                    gt,
                )

        except Exception as e:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
            log_reward(f"execution_reward 0 - exception {e=}", completion, gt)

    return rewards


def get_ngrams(tokens: List[str], n: int) -> set:
    """Generates a set of n-grams from a list of tokens."""
    # Ensure there are enough tokens to create at least one n-gram
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def n_gram_jaccard_similarity(candidate_query: str, gold_query: str, n: int) -> float:
    """Calculates the n-gram Jaccard similarity for a single n."""
    # Tokenize the SQL queries. Using .lower() for case-insensitivity.
    candidate_tokens = candidate_query.lower().split()
    gold_tokens = gold_query.lower().split()

    # Get the n-grams for both sets of tokens.
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    gold_ngrams = get_ngrams(gold_tokens, n)

    # Handle the edge case where one or both sets are empty.
    if not candidate_ngrams and not gold_ngrams:
        return 1.0
    if not candidate_ngrams or not gold_ngrams:
        return 0.0

    # Calculate Jaccard similarity.
    intersection = len(candidate_ngrams.intersection(gold_ngrams))
    union = len(candidate_ngrams.union(gold_ngrams))

    return intersection / union


def ensemble_n_gram_reward_func(completions, answer, **kwargs):
    """
    Calculates the averaged ensemble n-gram Jaccard similarity reward.
    This function computes the Jaccard similarity for n=1, 2, and 3
    and returns the average score for each sample.

    Args:
        completions (list[str]): Generated outputs
        answer (list[str]): Ground truth answers (from content with "role":"assistant" in train_text2sql_sft_dataset.json)

    Returns:
        list[float]: Reward scores
    """

    rewards = []
    questions = kwargs.get("question")
    evidences = kwargs.get("evidence")

    for completion, gt, question, evidence in zip(
        completions, answer, questions, evidences
    ):
        # print(f">>>>>ensemble_n_gram_reward_func: {gt=}")
        # print(f">>>>>ensemble_n_gram_reward_func: {completion=}")
        # print(f">>>>>ensemble_n_gram_reward_func: {question=}")
        # print(f">>>>>ensemble_n_gram_reward_func: {evidence=}")
        try:
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                log_reward("n_gram_reward 0 - no answer tag found", completion, gt)
                continue
            # Extract the "answer" part from the completion
            predicted_sql = match.group(1).strip()

            # Calculate Jaccard similarity for n=1, 2, and 3
            jaccard_1 = n_gram_jaccard_similarity(predicted_sql, gt, n=1)
            jaccard_2 = n_gram_jaccard_similarity(predicted_sql, gt, n=2)
            jaccard_3 = n_gram_jaccard_similarity(predicted_sql, gt, n=3)

            # Average the scores to get the final ensemble reward
            average_jaccard = (jaccard_1 + jaccard_2 + jaccard_3) / 3.0
            print(f"{average_jaccard=}")
            rewards.append(average_jaccard)
        except Exception as e:
            rewards.append(0.0)
            log_reward(f"n_gram_reward 0 - exception {e=}", completion, gt)

    return rewards


def llm_as_a_judge_reward_func(completions, answer, **kwargs):
    """
    Use Llama 3.3 70b as a judge to evaluate the quality of the generated SQL statements by comparing them to the ground truth answers.

    Args:
        completions (list[str]): Generated outputs
        answer (list[str]): Ground truth answers (from content with "role":"assistant" in train_text2sql_sft_dataset.json)

    Returns:
        list[float]: Reward scores
    """

    rewards = []

    client = Together()
    PROMPT_TEMPLATE = """
You are an experienced database expert. Your task is to evaluate a generated SQL query by comparing it
to the ground truth (gold) query and then assign a score between 0.0 and 2.0. A higher score indicates
the predicted query is more correct, while a score of 0.0 means it is completely incorrect.

Follow these evaluation rules strictly:

1. SELECT Clause:
• Only select columns that are mentioned in the user’s question.
• Do not include unnecessary columns or values.

2. Aggregation (MAX/MIN):
• Always perform JOINs before applying MAX() or MIN().

3. ORDER BY with Distinct Values:
• Use a GROUP BY <column> before an ORDER BY <column> ASC|DESC to ensure
distinct values.

4. Handling NULLs:
• If a column may contain NULL values (indicated by "None" in value examples
or explicitly mentioned), include a JOIN or a WHERE <column> IS NOT NULL
clause.

5. FROM/JOIN Clauses:
• Only include the tables essential for answering the question.

6. Strictly Follow Hints:
• Adhere to all hints provided with the question.

7. Thorough Question Analysis:
• Ensure all conditions and requirements mentioned in the question are ad-
dressed.

8. DISTINCT Keyword:
• Use SELECT DISTINCTwhen the question requires unique values (e.g., IDs, URLs)
or when column statistics (Value Statics) indicate its necessity.

9. Column Selection:
• Carefully analyze column descriptions and hints to choose the correct column
when similar columns exist across tables.

10. String Concatenation:
• Do not use any string concatenation methods (e.g., || ’ ’ ||) in the SELECT
clause.

11. JOIN Preference:
• Prefer using INNER JOINover nested SELECT statements.

12. Date Processing:
• Use STRFTIME()for any date manipulations (e.g., STRFTIME(’%Y’, SOMETIME)to
extract the year).

You are provided with the following inputs:
• Question: {QUESTION}
• Hint: {HINT}
• Gold Query: {GOLD_QUERY}
• Predicted Query: {PREDICTED_QUERY}

Based on the above, return a single numeric score between 0.0 and 2.0 that reflects how
correct the predicted query is compared to the gold query. Respond with only the score and
no additional explanation.
"""

    questions = kwargs.get("question")
    evidences = kwargs.get("evidence")
    for completion, gt, question, evidence in zip(
        completions, answer, questions, evidences
    ):
        try:
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                log_reward(
                    "llm_as_a_judge_reward_func 0 - no answer tag found", completion, gt
                )
                continue
            # Extract the "answer" part from the completion
            predicted_sql = match.group(1).strip()
            prompt = PROMPT_TEMPLATE.format(
                QUESTION=question,
                HINT=evidence,
                GOLD_QUERY=gt,
                PREDICTED_QUERY=predicted_sql,
            )
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            rewards.append(float(response.choices[0].message.content))
        except Exception as e:
            rewards.append(0.0)
            log_reward(f"llm_as_a_judge_reward_func 0 - exception {e=}", completion, gt)

    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


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
            # Format the rows as a simple table representation
            rows_prompt = "\n".join(
                "\t".join(str(val) for val in row) for row in values
            )
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


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = []
    SYSTEM_PROMPT = "You are a text to SQL query translator. Using the SQLite DB Schema and the External Knowledge, translate the following text question into a SQLite SQL select statement."

    input_json = json.load(open(TRAIN_JSON, "r"))

    for i, item in tqdm(enumerate(input_json)):
        print(f"processing #{i+1}")
        db_id = item["db_id"]
        question = item["question"]
        external_knowledge = item["evidence"]
        SQL = item["SQL"]
        db_path = DB_ROOT_PATH + "/" + db_id + "/" + db_id + ".sqlite"
        prompt = generate_combined_prompts_one(
            db_path,
            question,
            knowledge=external_knowledge,
        )

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": SQL + "\t----- bird -----\t" + db_id},
            ],
            "question": question,
            "evidence": external_knowledge,
        }

        ds.append(example)

    dataset_dict = {key: [d[key] for d in ds] for key in ds[0]}
    dataset = Dataset.from_dict(dataset_dict)

    def generate_r1_prompt(
        system_prompt, user_prompt, ground_truth, question, evidence
    ):
        r1_prefix = [
            {
                "role": "system",
                "content": """You are great at reasoning and translating natural language question to SQLite SQL query. Given DB Schema, External Knowledge, and Question, your task is to first generate step-by-step reasoning, then apply the resoning to generate the SQLite select statement as the accurate translation of the Question. Enclose the step-by-step reasoning within the <think> </think> tags, and the final SQL statement within the <answer> </answer> tags, i.e. <think> reasoning steps </think> <answer> final SQL </answer>.""",
            },
            {"role": "user", "content": user_prompt},
        ]

        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "answer": ground_truth,
            "question": question,
            "evidence": evidence,
        }

    # convert our dataset to the r1 prompt
    dataset = dataset.map(
        lambda x: generate_r1_prompt(
            x["messages"][0]["content"],
            x["messages"][1]["content"],
            x["messages"][2]["content"],
            x["question"],
            x["evidence"],
        ),
        remove_columns=dataset.column_names,  # Remove original columns to avoid conflicts with apply_chat_template
    )

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.3)

    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    print("len(train_dataset)", len(train_dataset))
    print(train_dataset[0])
    print("len(eval_dataset)", len(eval_dataset))
    print(eval_dataset[0])

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[
            format_reward_func,
            execution_reward_func,
            ensemble_n_gram_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.tokenizer = tokenizer

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    # JT: by default training_args.resume_from_checkpoint is None
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card({"tags": ["rl", "grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    # print("model_args", model_args)
    # print("script_args", script_args)
    # print("training_args", training_args)
    # exit()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

# two ways to run this script:
# with-proxy accelerate launch --num_processes 8 --config_file deepspeed_zero3.yaml grpo_text2sql.py --config grpo-llama323b-text2sql.yaml

# with-proxy nohup accelerate launch --num_processes 4 --config_file deepspeed_zero3.yaml grpo_text2sql.py --config grpo-llama323b-text2sql.yaml &
