from typing import Any, Dict, List

import pandas as pd


def calculate_relative_gain(tuned_score: float, baseline_score: float):
    baseline_score = max(baseline_score, 1e-2)
    return tuned_score / baseline_score - 1  # --> [-1, inf]


def calculate_transferability_index(eval_grid_results: List[Dict[str, Any]]):
    df = pd.DataFrame(eval_grid_results)

    # Flatten to list of models
    df_pivoted = (
        df.pivot(index="model", columns="task", values="score")
        .add_suffix("_score")
        .reset_index()
    )

    # Get all score columns dynamically
    score_columns = [col for col in df_pivoted.columns if col.endswith("_score")]

    # Get baseline scores
    baseline_scores = df_pivoted[df_pivoted["model"] == "base_model"][
        score_columns
    ].iloc[0]

    # Add relative gain columns
    for score_col in score_columns:
        task_name = score_col.replace("_score", "")
        gain_col = f"{task_name}_relative_gain"
        df_pivoted[gain_col] = df_pivoted.apply(
            lambda row: calculate_relative_gain(
                row[score_col], baseline_scores[score_col]
            ),
            axis=1,
        )

    df_pivoted["transferability"] = (
        df_pivoted["task2_relative_gain"] / df_pivoted["task1_relative_gain"]
    )

    return df_pivoted
