import typing as t

if t.TYPE_CHECKING:
    import dspy


class TaskDatasets(t.NamedTuple):
    trainset: t.Iterable["dspy.Example"]
    valset: t.Iterable["dspy.Example"]
    testset: t.Iterable["dspy.Example"]
