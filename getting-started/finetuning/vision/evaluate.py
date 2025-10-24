#!/usr/bin/env python3
"""
Script to evaluate a vision-language model on the W2 tax form dataset using compatible API client.
Leverages the OpenAI-compatible SDK for various endpoints, like vLLM server, Llama API, or any compatible API.
Support batch processing.
Loads images from the provided dataset, sends them to the compatible API server,
and compares with the expected output.
"""

import argparse
import base64
import json
import logging
import os
import pathlib
import re
import time
import traceback
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, load_from_disk
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class W2Form(BaseModel):
    box_b_employer_identification_number: str
    box_c_employer_name: str
    box_c_employer_street_address: str
    box_c_employer_city_state_zip: str
    box_a_employee_ssn: str
    box_e_employee_name: str
    box_e_employee_street_address: str
    box_e_employee_city_state_zip: str
    box_d_control_number: int
    box_1_wages: float
    box_2_federal_tax_withheld: float
    box_3_social_security_wages: float
    box_4_social_security_tax_withheld: float
    box_5_medicare_wages: float
    box_6_medicare_wages_tax_withheld: float
    box_7_social_security_tips: float
    box_8_allocated_tips: float
    box_9_advance_eic_payment: Optional[str]
    box_10_dependent_care_benefits: float
    box_11_nonqualified_plans: float
    box_12a_code: str
    box_12a_value: float
    box_12b_code: str
    box_12b_value: float
    box_12c_code: str
    box_12c_value: float
    box_12d_code: Optional[str]
    box_12d_value: float
    box_13_statutary_employee: Optional[str]
    box_13_retirement_plan: Optional[str]
    box_13_third_part_sick_pay: Optional[str]
    box_15_1_state: str
    box_15_1_employee_state_id: str
    box_16_1_state_wages: float
    box_17_1_state_income_tax: float
    box_18_1_local_wages: float
    box_19_1_local_income_tax: float
    box_20_1_locality: str
    box_15_2_state: str
    box_15_2_employee_state_id: str
    box_16_2_state_wages: float
    box_17_2_state_income_tax: float
    box_18_2_local_wages: float
    box_19_2_local_income_tax: float
    box_20_2_locality: str


# ----------- Utilities -----------
def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def create_messages(prompt: str, image_path: str) -> List[Dict]:
    """Create messages array for API client call."""
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"
            },
        },
    ]
    return [{"role": "user", "content": content}]


def clean_json_string(json_str: str) -> str:
    """
    Clean common JSON formatting issues from LLM responses.

    Args:
        json_str: Raw JSON string that may contain formatting issues

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code block markers
    json_str = re.sub(r"```(?:json)?\s*", "", json_str)
    json_str = re.sub(r"\s*```", "", json_str)

    # Fix malformed string patterns like: "field": ",\n" ,
    # This handles the specific error case where strings are malformed with newlines
    json_str = re.sub(r':\s*",\s*"\s*,', ': "",', json_str)

    # Fix incomplete string literals with control characters
    # Pattern: "field": "partial_value\nrest_of_value",
    json_str = re.sub(r':\s*"([^"]*)\n([^"]*)",', r': "\1\2",', json_str)

    # Fix the specific pattern from the error: "field": "value\n" followed by whitespace and comma
    json_str = re.sub(r':\s*"([^"]*)\n"\s*,', r': "\1",', json_str)

    # Remove trailing commas in objects and arrays
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # Fix missing quotes around keys (sometimes LLMs output unquoted keys)
    json_str = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)

    # Fix single quotes to double quotes (JSON requires double quotes)
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)

    # Remove control characters that are not allowed in JSON strings
    # Keep only printable ASCII and basic whitespace
    json_str = "".join(char for char in json_str if ord(char) >= 32 or char in "\t\r ")

    # Fix null-like values that should be proper JSON null
    json_str = re.sub(r":\s*None\s*,", ": null,", json_str, flags=re.IGNORECASE)
    json_str = re.sub(r":\s*undefined\s*,", ": null,", json_str, flags=re.IGNORECASE)

    return json_str


def extract_json_from_response(response: str) -> Tuple[Dict[str, Any], bool]:
    """
    Robust JSON extraction from LLM responses with comprehensive error handling.

    Args:
        response: Raw response text from LLM

    Returns:
        Tuple of (extracted_json_dict, has_error)
    """
    if not response or not response.strip():
        logger.warning("Empty response provided")
        return {}, True

    # Strategy 1: Look for JSON content between triple backticks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Strategy 2: Look for JSON object pattern (handle nested braces)
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Strategy 3: Find content between first { and last }
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
            else:
                logger.warning("No JSON pattern found in response")
                logger.debug(f"Response snippet: {response[:200]}...")
                return {}, True

    # Clean the extracted JSON string
    original_json_str = json_str
    json_str = clean_json_string(json_str)

    # Attempt to parse with multiple strategies
    parsing_strategies = [
        ("direct", lambda s: json.loads(s)),
        ("strip_whitespace", lambda s: json.loads(s.strip())),
        (
            "fix_escapes",
            lambda s: json.loads(s.replace("\\\\", "\\").replace('\\"', '"')),
        ),
    ]

    for strategy_name, parse_func in parsing_strategies:
        try:
            parsed_json = parse_func(json_str)

            # Validate that it's a dictionary (expected for most use cases)
            if not isinstance(parsed_json, dict):
                logger.warning(
                    f"Extracted JSON is not a dictionary: {type(parsed_json)}"
                )
                continue

            logger.debug(f"Successfully parsed JSON using strategy: {strategy_name}")
            return parsed_json, False

        except json.JSONDecodeError as e:
            logger.debug(f"Strategy '{strategy_name}' failed: {e}")
            continue
        except Exception as e:
            logger.debug(f"Unexpected error in strategy '{strategy_name}': {e}")
            continue

    # If all strategies fail, log details for debugging
    logger.error("All JSON parsing strategies failed")
    logger.debug(f"Original JSON string (first 500 chars): {original_json_str[:500]}")
    logger.debug(f"Cleaned JSON string (first 500 chars): {json_str[:500]}")

    return {}, True


def generate_prompt(structured=True) -> str:
    """Generate prompt for the model."""
    json_schema = W2Form.model_json_schema()

    prompt = (
        "You are an expert document information extraction system. "
        "I will show you an image of a W-2 tax form. "
        "Please extract all the information from this form and return it in a JSON format. "
        "Include all fields such as employee details, employer details, wages, federal income tax withheld, "
        "social security wages, social security tax withheld, medicare wages and tips, medicare tax withheld, "
        "and any other information present on the form. "
    )

    if not structured:
        prompt += f"Return ONLY the JSON output without any additional text or explanations following this schema {json_schema}"

    return prompt


def call_api_client(
    client: OpenAI,
    messages: List[Dict],
    model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    response_format: Optional[Dict] = None,
    timeout: int = 300,
    seed: Optional[int] = 42,
):
    """
    Call compatible API server using OpenAI-compatible client.
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        # Add seed if provided for reproducible generation
        if seed is not None:
            kwargs["seed"] = seed

        # Add response format if structured output is enabled
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"Making API client call with model: {model}")
        response = client.chat.completions.create(**kwargs)

        logger.debug(f"Received response with {len(response.choices)} choices")
        return response

    except Exception as e:
        logger.error(f"API client call failed: {e}")
        raise


def process_single_sample(
    client: OpenAI,
    sample_data: Tuple[int, Dict],
    output_dir: str,
    model: str,
    structured: bool,
    timeout: int,
) -> Dict[str, Any]:
    """Process a single sample using OpenAI SDK."""
    idx, sample = sample_data

    try:
        # Get image
        image = sample["image"]

        # Save image temporarily
        image_path = get_image_path(image, output_dir, idx)
        logger.debug(f"Saved image to {image_path}")

        # Generate prompt and messages
        prompt = generate_prompt(structured)
        messages = create_messages(prompt, image_path)

        # Prepare response format for structured output
        response_format = None
        if structured:
            json_schema = W2Form.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "W2Form",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # Call API client
        start_time = time.time()

        try:
            response = call_api_client(
                client=client,
                messages=messages,
                model=model,
                response_format=response_format,
                timeout=timeout,
            )

            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}

        except Exception as e:
            logger.error(f"Error calling SDK for sample {idx}: {e}")
            content = ""
            usage = {}

        processing_time = time.time() - start_time

        # Extract JSON from response
        extracted_json, json_parsing_error = extract_json_from_response(content)

        # Get ground truth
        ground_truth_raw = json.loads(sample["ground_truth"])

        # Handle the gt_parse wrapper structure if present
        if "gt_parse" in ground_truth_raw:
            ground_truth = ground_truth_raw["gt_parse"]
        else:
            ground_truth = ground_truth_raw

        # Normalize for comparison
        normalized_pred = normalize_json(extracted_json)
        normalized_gt = normalize_json(ground_truth)

        # Save results
        result = {
            "sample_id": idx,
            "prediction": extracted_json,
            "ground_truth": ground_truth,
            "normalized_prediction": normalized_pred,
            "normalized_gt": normalized_gt,
            "raw_response": content,
            "processing_time": processing_time,
            "json_parsing_error": json_parsing_error,
            "usage": usage,
        }

        return result

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Error processing sample {idx}: {str(e)} at line {traceback_str}")
        return {
            "sample_id": idx,
            "prediction": {},
            "ground_truth": {},
            "normalized_prediction": {},
            "normalized_gt": {},
            "raw_response": "",
            "processing_time": 0.0,
            "json_parsing_error": True,
            "usage": {},
            "error": str(e),
        }


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate accuracy metrics for the predictions."""
    if not results:
        logger.error("No results provided")
        return {"accuracy": 0.0, "field_accuracy": {}}

    # Initialize metrics
    total_fields = 0
    correct_fields = 0
    parse_errors = 0
    total_records = len(results)
    logger.info(f"Total records: {total_records}")
    field_counts = {}
    field_correct = {}

    for result in results:
        pred, gt = result["prediction"], result["ground_truth"]

        if result["json_parsing_error"]:
            parse_errors += 1
            total_fields += len(gt)
            continue

        for field in gt.keys():
            # Count total occurrences of this field
            field_counts[field] = field_counts.get(field, 0) + 1
            total_fields += 1

            # Check if field is correct
            if field in pred and pred[field] == gt[field]:
                correct_fields += 1
                field_correct[field] = field_correct.get(field, 0) + 1

    # Calculate overall accuracy
    accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
    errors = parse_errors / total_records if total_records > 0 else 0.0

    # Calculate per-field accuracy
    field_accuracy = {}
    for field in field_counts:
        field_accuracy[field] = field_correct.get(field, 0) / field_counts[field]

    return {
        "accuracy": accuracy,
        "field_accuracy": field_accuracy,
        "parse_error": errors,
    }


def normalize_field_value(value: Any) -> str:
    """Normalize field values for comparison."""
    if value is None:
        return ""

    # Convert to string and normalize
    value_str = str(value).strip().lower()

    # Remove common separators in numbers
    value_str = value_str.replace(",", "").replace(" ", "")

    # Try to convert to float for numeric comparison
    try:
        value_float = float(value_str)
        return str(value_float)
    except ValueError:
        return value_str


def normalize_json(json_obj: Dict) -> Dict:
    """Normalize JSON object for comparison."""
    normalized = {}

    for key, value in json_obj.items():
        # Normalize key (lowercase, remove spaces)
        norm_key = key.lower().replace(" ", "_")

        # Normalize value
        if isinstance(value, dict):
            normalized[norm_key] = normalize_json(value)
        elif isinstance(value, list):
            normalized[norm_key] = [normalize_field_value(v) for v in value]
        else:
            normalized[norm_key] = normalize_field_value(value)

    return normalized


def get_image_path(image: Image.Image, output_dir: str, idx: int) -> str:
    """Get the path to save the image."""
    # Create a temporary file for the image
    temp_dir = pathlib.Path(output_dir) / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = temp_dir / f"temp_{idx}.png"
    image_path = str(image_path.resolve())
    image.save(image_path)
    return image_path


def vllm_openai_sdk_evaluation(
    test_set,
    output_dir: str,
    server_url: str = "http://localhost:8001",
    api_key: str = "default-blank-localhost",
    model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    structured: bool = True,
    timeout: int = 300,
    max_workers: int = 10,
):
    """
    Evaluate the W2 extraction task using OpenAI SDK with batch processing.
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,  # vLLM doesn't require a real API key
        base_url=f"{server_url}",
    )

    # Prepare sample data for batch processing
    sample_data = [(idx, sample) for idx, sample in enumerate(test_set)]

    results = []

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(
                process_single_sample,
                client,
                data,
                output_dir,
                model,
                structured,
                timeout,
            ): data[0]
            for data in sample_data
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_sample),
            total=len(sample_data),
            desc="Processing samples (batch)",
        ):
            sample_idx = future_to_sample[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Exception in sample {sample_idx}: {e}")
                # Add error result
                results.append(
                    {
                        "sample_id": sample_idx,
                        "prediction": {},
                        "ground_truth": {},
                        "normalized_prediction": {},
                        "normalized_gt": {},
                        "raw_response": "",
                        "processing_time": 0.0,
                        "json_parsing_error": True,
                        "usage": {},
                        "error": str(e),
                    }
                )

    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x["sample_id"])

    return results


def vllm_openai_sdk_sequential_evaluation(
    test_set,
    output_dir: str,
    server_url: str = "http://localhost:8001",
    api_key: str = "default-blank-localhost",
    model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    structured: bool = True,
    timeout: int = 300,
):
    """
    Evaluate the W2 extraction task using OpenAI SDK sequentially (for debugging).
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,  # vLLM doesn't require a real API key
        base_url=f"{server_url}",
    )

    results = []

    for idx, sample in enumerate(
        tqdm(test_set, desc="Processing samples (sequential)")
    ):
        result = process_single_sample(
            client, (idx, sample), output_dir, model, structured, timeout
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language model on W2 tax form dataset"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8001",
        help="URL of the vLLM HTTP server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="singhsays/fake-w2-us-tax-form-dataset",
        help="Name of the Huggingface dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./w2_evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10, use -1 for all)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        default=False,
        help="Whether to use structured output (JSON schema)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for SDK requests in seconds",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers for batch processing",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Process samples sequentially instead of in parallel (for debugging)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    test_set = None
    if Path(args.dataset_name, "state.json").exists():
        test_set = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)
        if "test" not in dataset:
            logger.error("Dataset does not have a test split")
            return 1
        test_set = dataset["test"]

    logger.info(f"Loaded test set with {len(test_set)} samples")

    # Limit number of samples if specified
    if args.limit > 0 and args.limit < len(test_set):
        test_set = test_set.select(range(args.limit))
        logger.info(f"Limited to {args.limit} samples")

    # Get API key from environment variable
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.warning(
            "No API key found. Please set the TOGETHER_API_KEY or OPENAI_API_KEY environment variable for public APIs."
        )
        api_key = "default-blank-localhost"

    # Test server connection
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=f"{args.server_url}",
        )
        # Test with a simple call
        # models = client.models.list()
        logger.info(f"Successfully connected to server at {args.server_url}")
        # logger.info(f"Available models: {[model.id for model in models.data]}")
    except Exception as e:
        logger.error(f"Failed to connect to server at {args.server_url}: {e}")
        logger.error("Make sure the server is running and accessible")
        return 1

    # Run evaluation
    if args.sequential:
        logger.info("Running sequential evaluation...")
        results = vllm_openai_sdk_sequential_evaluation(
            test_set=test_set,
            output_dir=args.output_dir,
            server_url=args.server_url,
            api_key=api_key,
            model=args.model,
            structured=args.structured,
            timeout=args.timeout,
        )
    else:
        logger.info(f"Running batch evaluation with {args.max_workers} workers...")
        results = vllm_openai_sdk_evaluation(
            test_set=test_set,
            output_dir=args.output_dir,
            server_url=args.server_url,
            api_key=api_key,
            model=args.model,
            structured=args.structured,
            timeout=args.timeout,
            max_workers=args.max_workers,
        )

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        results_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {results_file}")

    except Exception as e:
        logger.error(f"Error saving detailed results: {str(e)}")
        return 1

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Save evaluation summary
    output_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    arguments = {
        "server_url": args.server_url,
        "model": args.model,
        "output_dir": args.output_dir,
        "dataset_name": args.dataset_name,
        "limit": args.limit,
        "structured": args.structured,
        "timeout": args.timeout,
        "max_workers": args.max_workers,
        "sequential": args.sequential,
        "prompt": generate_prompt(args.structured),
    }

    summary = {
        "arguments": arguments,
        "metrics": metrics,
        "timestamp": timestamp,
        "total_samples": len(results),
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Overall accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Parse error rate: {metrics['parse_error']:.4f}")
    logger.info("Field-level accuracy:")
    field_accuracy = metrics["field_accuracy"]
    for field, acc in sorted(field_accuracy.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {field}: {acc:.4f}")

    logger.info(f"Results saved to {output_file}")

    # Clean up temp directory if it exists
    temp_dir = os.path.join(args.output_dir, "temp")
    if os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir)

    return 0


if __name__ == "__main__":
    exit(main())
