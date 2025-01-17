import typing as t

import dspy

from datasets import load_dataset

from .datatypes import TaskDatasets
from .helpers import train_val_test_split


def datasets(
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    """
    Load dataset, dataset should be datasets.Dataset type (NOT DatasetDict, OR split the dataset yourself how you want)
    """
    dataset = load_dataset(
        "meta-llama/Llama-3.3-70B-Instruct-evals",
        "Llama-3.3-70B-Instruct-evals__mmlu_pro__details",
    )
    return train_val_test_split(
        dataset["latest"],
        _task_doc_example,
        train_size,
        validation_size,
    )


class TaskDoc(t.TypedDict):
    task_type: str
    task_name: str
    subtask_name: str
    input_question: str
    input_choice_list: dict
    input_final_prompts: list
    input_correct_responses: list
    output_prediction_text: list
    output_parsed_answer: str
    output_choice_completions: t.Optional[dict]
    output_choice_negative_log_likelihoods: t.Optional[dict]
    output_metrics: dict
    is_correct: bool
    input_question_hash: str
    input_final_prompts_hash: list
    benchmark_label: str
    eval_config: dict


inputs = ["input_question", "input_choice_list"]
outputs = ["output_parsed_answer"]


def _task_doc_example(doc: TaskDoc) -> dspy.Example:
    example = dspy.Example(
        question=doc["input_question"],
        options=doc["input_choice_list"],
        answer=doc["output_parsed_answer"],
    )
    example._input_keys = {"question", "options"}
    example._output_keys = {"answer"}
    return example


def signature(instructions: str = "") -> dspy.Signature:
    class MMLUPro(dspy.Signature):
        __doc__ = instructions
        question: str = dspy.InputField(desc="The question to be answered")
        options: dict = dspy.InputField(desc="Dictionary of answer choices")
        answer: str = dspy.OutputField(desc="The correct answer letter")

    return MMLUPro


def metric(gold: dspy.Example, pred: dspy.Example, trace=False) -> bool:
    return gold.answer == pred.answer
