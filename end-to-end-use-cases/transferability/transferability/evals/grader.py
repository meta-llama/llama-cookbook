import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel

from ..utils import map_with_progress

Row = dict[str, Any]


class EvalResult(BaseModel):
    rows: list[Row]  # raw rows
    # overall metrics
    metrics: dict[str, float] | None = None
    # result for each row
    result_data: list[dict[str, Any]]
    topline_metric_name: str


class IGrader(ABC):
    @abstractmethod
    def grade_row(self, row: Row) -> dict[str, Any]:
        """Calculates metrics for a single row."""
        pass

    @abstractmethod
    def calculate_aggregate_metrics(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculates aggregate metrics for a list of row results.
        This is used for both overall and per-subset calculations.

        :param results: List of row results, returned by grade_row
        :param rows: List of input rows
        """
        pass

    @abstractmethod
    def topline_metric(self) -> str:
        """Key of the grade value in the overall metrics dict."""
        pass

    def grade(self, rows: list[Row]) -> EvalResult:
        """Grades rows, calculating overall and per-subset metrics using helper methods."""

        result_data = map_with_progress(self.grade_row, rows)
        overall_metrics = self.calculate_aggregate_metrics(result_data)

        return EvalResult(
            rows=rows,
            metrics=overall_metrics,
            result_data=result_data,
            topline_metric_name=self.topline_metric(),
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class JSONGrader(IGrader):
    """Generic grader for JSON-based structured data extraction tasks.

    This grader can be used for any task where the model outputs JSON that needs
    to be compared against ground truth JSON, such as W2 form extraction, OCR tasks, etc.
    """

    def grade_row(self, row: Row) -> dict[str, Any]:
        from .json_grading_utils import calculate_json_accuracy, JSONUtils

        ground_truth = JSONUtils.load_json_from_str(row["expected_output"])
        if "gt_parse" in ground_truth:
            ground_truth = ground_truth["gt_parse"]
        json_response = JSONUtils.extract_json_from_response(row["raw_response"])
        return calculate_json_accuracy(ground_truth, json_response)

    def calculate_aggregate_metrics(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate aggregate metrics for JSON-based evaluation."""
        if not results:
            return {"accuracy": 0.0, "error_rate": 1.0}

        results_df = pd.DataFrame(results)

        # Calculate accuracy (mean of scores)
        accuracy = results_df["score"].mean() if "score" in results_df.columns else 0.0

        # Calculate error rate (percentage of failed parsing attempts)
        error_rate = (
            (results_df["score"] == -1).mean() if "score" in results_df.columns else 0.0
        )

        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "total_samples": len(results),
        }

    def topline_metric(self) -> str:
        return "accuracy"


# Grader Registry - Clean factory pattern
GRADER_REGISTRY = {
    "JSONGrader": JSONGrader,
}


def get_grader(grader_name: str):
    """Factory function to get grader by name from config."""
    if grader_name not in GRADER_REGISTRY:
        available = ", ".join(GRADER_REGISTRY.keys())
        raise ValueError(f"Unknown grader '{grader_name}'. Available: {available}")
    return GRADER_REGISTRY[grader_name]()
