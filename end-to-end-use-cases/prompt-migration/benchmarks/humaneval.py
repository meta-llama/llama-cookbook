import typing as t

from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from datasets import load_dataset
from lm_eval.evaluator_utils import eval_logger
import dspy

from .datatypes import TaskDatasets
from .helpers import train_val_test_split

if t.TYPE_CHECKING:
    from bigcode_eval.base import Task


def signature(instructions: str = "") -> dspy.Signature:
    class HumanEval(dspy.Signature):
        __doc__ = instructions
        prompt: str = dspy.InputField()
        solution: str = dspy.OutputField()

    return HumanEval


def metric(gold: dspy.Example, pred: dspy.Example, trace=False) -> bool:
    program = gold.prompt + "\n" + pred.solution + "\n" + gold.dspy_test
    result = check_correctness(
        program,
        timeout=30,
        task_id=gold.dspy_task_id,
        completion_id=None,
    )

    if result["passed"]:
        return True

    eval_logger.debug(f"{gold.dspy_task_id}: {result['result']}")
    return False


def datasets(
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    dataset = load_dataset("codeparrot/instructhumaneval")
    train_docs, validation_docs, test_docs = train_val_test_split(
        dataset,
        train_size=train_size,
        validation_size=validation_size,
    )

    return TaskDatasets(
        trainset=map(_task_doc_example, train_docs),
        valset=map(_task_doc_example, validation_docs),
        testset=map(_task_doc_example, test_docs),
    )


class TaskDoc(t.TypedDict):
    task_id: str
    prompt: str
    canonical_solution: str
    test: str


inputs = ["prompt"]
outputs = ["solution"]


def _task_doc_example(doc: TaskDoc) -> dspy.Example:
    return dspy.Example(
        prompt=doc["prompt"],
        solution=doc["canonical_solution"],
        # dspy_ keys are hidden
        dspy_task_id=doc["task_id"],
        dspy_test=doc["test"],
    ).with_inputs(*inputs)