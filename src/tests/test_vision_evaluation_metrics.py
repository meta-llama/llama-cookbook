# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure vision evaluation module can be imported even in lightweight test environments
for mod_name in ["datasets", "openai", "PIL", "pydantic", "tqdm"]:
    if mod_name not in sys.modules:
        if mod_name == "pydantic":
            mock_pydantic = MagicMock()
            class DummyBaseModel:
                @classmethod
                def model_json_schema(cls):
                    props = {k: {} for k in getattr(cls, "__annotations__", {}).keys()}
                    return {"properties": props}
            mock_pydantic.BaseModel = DummyBaseModel
            sys.modules["pydantic"] = mock_pydantic
        else:
            sys.modules[mod_name] = MagicMock()

import importlib.util
from pathlib import Path

eval_path = Path(__file__).resolve().parents[2] / "getting-started" / "finetuning" / "vision" / "evaluate.py"
spec = importlib.util.spec_from_file_location("vision_evaluate", eval_path)
vision_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision_eval)


def test_extract_ground_truth_from_dict():
    sample = {"ground_truth": {"field_a": "1", "field_b": "2"}}
    gt = vision_eval._extract_ground_truth(sample)
    assert gt == {"field_a": "1", "field_b": "2"}


def test_extract_ground_truth_from_json_string():
    sample = {"ground_truth": json.dumps({"field_a": "1", "field_b": "2"})}
    gt = vision_eval._extract_ground_truth(sample)
    assert gt == {"field_a": "1", "field_b": "2"}


def test_extract_ground_truth_nested_gt_parse():
    sample = {"ground_truth": json.dumps({"gt_parse": {"box_1": 100.0, "box_2": 20.0}})}
    gt = vision_eval._extract_ground_truth(sample)
    assert gt == {"box_1": 100.0, "box_2": 20.0}


def test_extract_ground_truth_fallback_schema():
    sample = {"ground_truth": "invalid_json"}
    gt = vision_eval._extract_ground_truth(sample)
    assert len(gt) > 0
    assert "box_1_wages" in gt


def test_calculate_metrics_includes_failed_samples_in_denominator():
    # Sample 1: Successful prediction (10 fields, 10 correct)
    sample_1 = {
        "sample_id": 0,
        "prediction": {"f1": "a", "f2": "b"},
        "ground_truth": {"f1": "a", "f2": "b"},
        "json_parsing_error": False,
    }
    # Sample 2: Crashed sample with ground truth preserved (2 fields)
    sample_2 = {
        "sample_id": 1,
        "prediction": {},
        "ground_truth": {"f1": "a", "f2": "b"},
        "json_parsing_error": True,
        "error": "Image read failure",
    }

    metrics = vision_eval.calculate_metrics([sample_1, sample_2])
    # Total fields should be 2 + 2 = 4, correct fields = 2
    # Accuracy should be 2 / 4 = 0.50 (not 2 / 2 = 1.0)
    assert metrics["accuracy"] == 0.5
    assert metrics["parse_error"] == 0.5


def test_process_single_sample_exception_preserves_ground_truth():
    mock_client = MagicMock()
    sample_data = (
        0,
        {"ground_truth": json.dumps({"box_1": "test"}), "image": None},
    )

    with patch.object(vision_eval, "get_image_path", side_effect=RuntimeError("Corrupt image")):
        result = vision_eval.process_single_sample(
            client=mock_client,
            sample_data=sample_data,
            output_dir="/tmp",
            model="test-model",
            structured=False,
            timeout=10,
        )

    assert result["json_parsing_error"] is True
    assert result["ground_truth"] == {"box_1": "test"}
    assert result["sample_id"] == 0


def test_process_single_sample_dict_ground_truth_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"box_1": "test"}'
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    sample_data = (
        0,
        {"ground_truth": {"box_1": "test"}, "image": "dummy.png"},
    )

    with patch.object(vision_eval, "get_image_path", return_value="/tmp/dummy.png"), \
         patch.object(vision_eval, "call_api_client", return_value=mock_response):
        result = vision_eval.process_single_sample(
            client=mock_client,
            sample_data=sample_data,
            output_dir="/tmp",
            model="test-model",
            structured=False,
            timeout=10,
        )

    assert result["json_parsing_error"] is False
    assert result["ground_truth"] == {"box_1": "test"}
    assert result["prediction"] == {"box_1": "test"}


if __name__ == "__main__":
    test_extract_ground_truth_from_dict()
    test_extract_ground_truth_from_json_string()
    test_extract_ground_truth_nested_gt_parse()
    test_extract_ground_truth_fallback_schema()
    test_calculate_metrics_includes_failed_samples_in_denominator()
    test_process_single_sample_exception_preserves_ground_truth()
    test_process_single_sample_dict_ground_truth_success()
    print("All vision evaluation metrics tests passed!")
