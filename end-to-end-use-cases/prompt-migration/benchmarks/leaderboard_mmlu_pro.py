import typing as t

from datasets import load_dataset
import dspy

from .datatypes import TaskDatasets
from .helpers import train_val_test_split


def signature(instructions: str = "") -> dspy.Signature:
    class MMLUPro(dspy.Signature):
        __doc__ = instructions
        question: str = dspy.InputField()
        options: list[str] = dspy.InputField()
        answer: str = dspy.OutputField()

    return MMLUPro


def metric(gold: dspy.Example, pred: dspy.Example, trace=False) -> bool:
    return gold.answer == pred.answer


def datasets(
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    return train_val_test_split(
        dataset["test"], _task_doc_example, train_size, validation_size
    )


class TaskDoc(t.TypedDict):
    question_id: int
    question: str
    options: list[str]
    answer: str
    answer_index: int
    cot_content: str
    category: str
    src: str


inputs = ["question", "options"]
outputs = ["answer"]


def _num_letter(n: int) -> str:
    return chr(ord("A") + n)


def _task_doc_example(doc: TaskDoc) -> dspy.Example:
    question = doc["question"]
    options = [f"{_num_letter(i)}. {option}" for i, option in enumerate(doc["options"])]
    answer = doc["answer"]
    return dspy.Example(
        question=question,
        options=options,
        answer=answer,
    ).with_inputs(*inputs)
