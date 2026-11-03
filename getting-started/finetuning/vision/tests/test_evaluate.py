import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def evaluate():
    spec = importlib.util.spec_from_file_location(
        "vision_evaluate", Path(__file__).resolve().parents[1] / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wrapped", [False, True])
def test_image_failure_preserves_ground_truth_and_counts_as_wrong(evaluate, wrapped):
    ground_truth = {
        "box_b_employer_identification_number": "123",
        "box_c_employer_name": "ACME",
    }
    sample = {
        "ground_truth": json.dumps(
            {"gt_parse": ground_truth} if wrapped else ground_truth
        )
    }
    result = evaluate.process_single_sample(None, (1, sample), ".", "test", False, 1)
    assert result["json_parsing_error"]
    assert result["ground_truth"] == ground_truth
    success = {
        "prediction": ground_truth,
        "ground_truth": ground_truth,
        "json_parsing_error": False,
    }
    metrics = evaluate.calculate_metrics([success, result])
    assert metrics["accuracy"] == 0.5
    assert metrics["parse_error"] == 0.5


@pytest.mark.parametrize("wrapped", [False, True])
def test_image_failure_with_preparsed_dict_ground_truth(evaluate, wrapped):
    # A dataset whose ground_truth column is already a dict (no JSON
    # stringification) must still keep its labels on the failure path
    # instead of being replaced by the full expected-field schema.
    ground_truth = {
        "box_b_employer_identification_number": "123",
        "box_c_employer_name": "ACME",
    }
    sample = {"ground_truth": {"gt_parse": ground_truth} if wrapped else ground_truth}
    result = evaluate.process_single_sample(None, (1, sample), ".", "test", False, 1)
    assert result["json_parsing_error"]
    assert result["ground_truth"] == ground_truth
    success = {
        "prediction": ground_truth,
        "ground_truth": ground_truth,
        "json_parsing_error": False,
    }
    metrics = evaluate.calculate_metrics([success, result])
    assert metrics["accuracy"] == 0.5
    assert metrics["parse_error"] == 0.5


def test_executor_failure_preserves_sample_ground_truth(evaluate, monkeypatch):
    ground_truth = {"box_c_employer_name": "ACME"}
    sample = {"ground_truth": json.dumps(ground_truth)}

    def crash(*args, **kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(evaluate, "process_single_sample", crash)
    results = evaluate.vllm_openai_sdk_evaluation([sample], ".", max_workers=1)
    assert results[0]["ground_truth"] == ground_truth
    assert results[0]["json_parsing_error"]


@pytest.mark.parametrize("raw", ["not JSON", "[]", '"text"', "{}"])
def test_unusable_ground_truth_counts_expected_form_fields(evaluate, raw):
    result = evaluate.process_single_sample(
        None, (0, {"ground_truth": raw}), ".", "test", False, 1
    )
    assert result["json_parsing_error"]
    assert set(result["ground_truth"]) == set(
        evaluate.W2Form.model_json_schema()["properties"]
    )
    success = {
        "prediction": {"field": "ok"},
        "ground_truth": {"field": "ok"},
        "json_parsing_error": False,
    }
    metrics = evaluate.calculate_metrics([success, result])
    assert metrics["accuracy"] == 1 / (1 + len(result["ground_truth"]))


def test_executor_failure_without_ground_truth_uses_expected_fields(
    evaluate, monkeypatch
):
    def crash(*args, **kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(evaluate, "process_single_sample", crash)
    result = evaluate.vllm_openai_sdk_evaluation([{}], ".", max_workers=1)[0]
    assert set(result["ground_truth"]) == set(
        evaluate.W2Form.model_json_schema()["properties"]
    )


def test_successful_sample_behavior_is_unchanged(evaluate, monkeypatch):
    from types import SimpleNamespace

    ground_truth = {"box_c_employer_name": "ACME"}
    monkeypatch.setattr(evaluate, "get_image_path", lambda *args: "image.png")
    monkeypatch.setattr(evaluate, "create_messages", lambda *args: [])
    monkeypatch.setattr(
        evaluate,
        "call_api_client",
        lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(ground_truth))
                )
            ],
            usage=None,
        ),
    )
    result = evaluate.process_single_sample(
        None,
        (
            0,
            {"image": object(), "ground_truth": json.dumps({"gt_parse": ground_truth})},
        ),
        ".",
        "test",
        False,
        1,
    )
    assert result["ground_truth"] == ground_truth
    assert not result["json_parsing_error"]
    assert evaluate.calculate_metrics([result])["accuracy"] == 1.0
