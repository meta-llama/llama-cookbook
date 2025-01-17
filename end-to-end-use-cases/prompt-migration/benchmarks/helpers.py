import typing as t

from .datatypes import TaskDatasets

if t.TYPE_CHECKING:
    from datasets import Dataset
    import dspy


def train_val_test_split(
    dataset: "Dataset",
    mapper: t.Callable[[dict], "dspy.Example"],
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    docs = dataset.train_test_split(train_size=train_size)
    train_docs = docs["train"]
    docs = docs["test"].train_test_split(train_size=validation_size)
    validation_docs = docs["train"]
    test_docs = docs["test"]
    return TaskDatasets(
        trainset=list(map(mapper, train_docs)),
        valset=list(map(mapper, validation_docs)),
        testset=list(map(mapper, test_docs)),
    )
